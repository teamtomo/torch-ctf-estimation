"""Estimate CTF from a 2D or 3D image."""

from typing import Any, Optional

import einops
import torch
from torch_fourier_rescale import fourier_rescale_2d
from torch_grid_utils.patch_grid import patch_grid

from torch_ctf_estimation.estimate_ctf_1d import (
    estimate_ctf_1d,
    fit_background_spline_1d,
)
from torch_ctf_estimation.estimate_ctf_2d import estimate_ctf_2d
from torch_ctf_estimation.estimate_ctf_2d.estimate_background_2d import (
    estimate_background_2d,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
    _estimate_defocus_2d_at_1x1,
)
from torch_ctf_estimation.models import (
    CTFFittingParams,
    Defocus1DResults,
    Defocus2DResults,
    LaserParams,
    OpticalParams,
    linear_tilt_axis_and_magnitude_deg,
)
from torch_ctf_estimation.utils.data_io import write_results_json
from torch_ctf_estimation.utils.defocus_field_from_1d import (
    _defocus_field_from_1d_fits,
)
from torch_ctf_estimation.utils.normalize import normalize_image


def estimate_ctf(
    image: torch.Tensor,  # (t, h, w) or (h, w)
    optical_params: OpticalParams,
    fitting_params: CTFFittingParams,
    laser_params: LaserParams | None = None,
    device: torch.device | None = None,
    results_path: str | None = None,
) -> tuple[torch.Tensor, Defocus1DResults, Defocus2DResults]:
    """
    Estimate CTF from a 2D or 3D image.

    Parameters
    ----------
    image : torch.Tensor
        (t, h, w) or (h, w) array containing 2D or 3D image data.
    optical_params : OpticalParams
        Pixel spacing, voltage, Cs, amplitude contrast.
    fitting_params : CTFFittingParams
        Defocus grid resolution, frequency range, patch size, and fitting options.
    laser_params : LaserParams | None, optional
        If set, use LPP CTF model for 2D estimation; if None, use standard CTF.
    device : torch.device | None, optional
        Device for computation. If None, uses cuda:0 when available, else cpu.
    results_path : str | None, optional
        If set, write hierarchical results (defocus, phase shift, B envelope)
        to this path.

    Returns
    -------
    mean_ps : torch.Tensor
        Mean power spectrum of the patches.
    result1d : Defocus1DResults
        Results from 1D defocus estimation.
    result2d : Defocus2DResults
        Results from 2D defocus estimation.
    """
    # -------------------------------------------------------------------------
    # Step 1: Setup — device, normalize image, optional rescaling
    # -------------------------------------------------------------------------
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.float().to(device)

    # Ensure shape (t, h, w); single 2D image becomes (1, h, w)
    image, _ = einops.pack([image], pattern="* h w")

    image = normalize_image(image)
    new_spacing = max(3.0, optical_params.pixel_spacing_angstroms)
    image, _ = fourier_rescale_2d(
        image=image,
        source_spacing=optical_params.pixel_spacing_angstroms,
        target_spacing=new_spacing,
    )
    t, h, w = image.shape

    # -------------------------------------------------------------------------
    # Step 2: Patch extraction — whole image or overlapping patches
    # -------------------------------------------------------------------------
    use_whole_image = fitting_params.patch_sidelength < 0
    if not use_whole_image and (
        h < fitting_params.patch_sidelength or w < fitting_params.patch_sidelength
    ):
        raise ValueError(
            f"Rescaled image size ({h}, {w}) is smaller than patch_sidelength "
            f"({fitting_params.patch_sidelength}). Use a larger image, a smaller "
            "patch_sidelength, or less aggressive rescaling (e.g. "
            "pixel_spacing_angstroms closer to the internal target spacing)."
        )

    if use_whole_image:
        print("Using whole-image mode")
        nt, nh, nw = fitting_params.defocus_grid_resolution
        if nh != 1 or nw != 1:
            raise ValueError(
                "When using whole-image mode (patch_sidelength < 0), "
                "defocus_grid_resolution must have nh=1 and nw=1, "
                f"got (nt={nt}, nh={nh}, nw={nw})."
            )

    if use_whole_image:
        patches, patch_centers = patch_grid(
            images=image,
            patch_shape=(1, h, w),
            patch_step=(1, h, w),
        )
        image_sidelength_for_1d = min(h, w)
    else:
        patches, patch_centers = patch_grid(
            images=image,
            patch_shape=(
                1,
                fitting_params.patch_sidelength,
                fitting_params.patch_sidelength,
            ),
            patch_step=(
                1,
                fitting_params.patch_sidelength // 2,
                fitting_params.patch_sidelength // 2,
            ),
        )
        image_sidelength_for_1d = fitting_params.patch_sidelength

    patches = einops.rearrange(patches, "t gh gw 1 ph pw -> t gh gw ph pw")

    # -------------------------------------------------------------------------
    # Step 3: Power spectra — FFT of each patch and global mean
    # -------------------------------------------------------------------------
    patch_ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2
    mean_ps = einops.reduce(patch_ps, "... ph pw -> ph pw", reduction="mean")

    # Decide whether to build 2D defocus from per-patch 1D fits (use_1d_spatial)
    # or from a single 2D optimisation over the full grid.
    nt, nh, nw = fitting_params.defocus_grid_resolution
    use_1d_spatial = fitting_params.use_1d_defocus_for_spatial and (
        fitting_params.defocus_model == "linear" or (nh > 1 or nw > 1)
    )
    # If using per-patch 1D: fit background spline once on mean spectrum, reuse for all
    bg_mean: Optional[Any] = None
    if use_1d_spatial:
        bg_mean = fit_background_spline_1d(
            power_spectrum=mean_ps,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            pixel_spacing_angstroms=new_spacing,
        )

    # -------------------------------------------------------------------------
    # Step 4: 1D CTF estimation — defocus (and optional B, phase) from mean spectrum
    # -------------------------------------------------------------------------
    result1d = estimate_ctf_1d(
        power_spectrum=mean_ps,
        image_sidelength=image_sidelength_for_1d,
        frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
        defocus_range_microns=fitting_params.defocus_range_microns,
        voltage_kev=optical_params.voltage_kev,
        spherical_aberration_mm=optical_params.spherical_aberration_mm,
        amplitude_contrast=optical_params.amplitude_contrast_fraction,
        pixel_spacing_angstroms=new_spacing,
        optimize_envelope=fitting_params.optimize_envelope_1d,
        b_range=fitting_params.b_range_1d,
        b_step=fitting_params.b_step_1d,
        refine_steps=fitting_params.refine_steps_1d,
        background_result=bg_mean,
        optimize_phase_shift=fitting_params.optimize_phase_shift,
        initial_phase_shift=fitting_params.initial_phase_shift,
    )

    # -------------------------------------------------------------------------
    # Step 5: 2D background — estimate and subtract before 2D defocus fit
    # -------------------------------------------------------------------------
    image_shape_2d = (
        (h, w)
        if use_whole_image
        else (image_sidelength_for_1d, image_sidelength_for_1d)
    )
    background_2d = estimate_background_2d(
        power_spectrum=mean_ps,
        image_sidelength=image_shape_2d,
    )
    patch_ps -= background_2d

    # -------------------------------------------------------------------------
    # Step 6: Prepare 2D fit — normalised positions, initial defocus/B/phase from 1D
    # -------------------------------------------------------------------------
    image_dimension_lengths = (
        torch.tensor([t - 1, h - 1, w - 1]).float().to(patch_ps.device)
    )
    normalised_patch_positions = patch_centers / image_dimension_lengths

    initial_envelope_B_2d = fitting_params.initial_envelope_B
    if initial_envelope_B_2d is None and result1d.ctf_model.envelope_B is not None:
        # use 1D-estimated B for 2D envelope
        if isinstance(result1d.ctf_model.envelope_B, torch.Tensor):
            initial_envelope_B_2d = float(
                result1d.ctf_model.envelope_B.detach().cpu().item()
            )
        else:
            initial_envelope_B_2d = float(result1d.ctf_model.envelope_B)
    if initial_envelope_B_2d is None:
        initial_envelope_B_2d = 0.0

    initial_defocus_2d = result1d.ctf_model.defocus_um
    if isinstance(initial_defocus_2d, torch.Tensor):
        initial_defocus_2d = float(initial_defocus_2d.cpu().item())
    else:
        initial_defocus_2d = float(initial_defocus_2d)

    initial_phase_shift_2d = fitting_params.initial_phase_shift
    if (
        fitting_params.optimize_phase_shift
        and result1d.ctf_model.phase_shift_degrees is not None
    ):
        p = result1d.ctf_model.phase_shift_degrees
        initial_phase_shift_2d = (
            float(p.cpu().item()) if isinstance(p, torch.Tensor) else float(p)
        )

    # -------------------------------------------------------------------------
    # Branch A: Build 2D defocus from per-patch 1D fits (grid or linear over space)
    # -------------------------------------------------------------------------
    if use_1d_spatial:
        # 2D fit at center only (1x1) to get astigmatism and center defocus
        result_1x1 = _estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=fitting_params.optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            n_iterations=fitting_params.n_iterations_2d,
            debug=fitting_params.debug,
            optimize_phase_shift=fitting_params.optimize_phase_shift,
            phase_shift_model=fitting_params.phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=fitting_params.phase_shift_quadratic_perpendicular_axis,
            initial_phase_shift=initial_phase_shift_2d,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
        )
        # Per-patch 1D defocus, then fit grid or linear model to those values
        result2d = _defocus_field_from_1d_fits(
            patch_power_spectra=patch_ps,
            normalised_patch_positions=normalised_patch_positions,
            result_1x1=result_1x1,
            defocus_model=fitting_params.defocus_model,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            initial_defocus=initial_defocus_2d,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            defocus_range_microns=fitting_params.defocus_range_microns,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            pixel_spacing_angstroms=new_spacing,
            optimize_envelope_1d=fitting_params.optimize_envelope_1d,
            b_range_1d=fitting_params.b_range_1d,
            b_step_1d=fitting_params.b_step_1d,
            refine_steps_1d=fitting_params.refine_steps_1d,
            background_result=bg_mean,
            device=patch_ps.device,
            optimize_phase_shift=fitting_params.optimize_phase_shift,
        )
        # For linear defocus: compute tilt axis and magnitude (degrees) for reporting
        if result2d.defocus_model_type == "linear":
            axis_deg, tilt_deg = linear_tilt_axis_and_magnitude_deg(
                result2d, new_spacing, min(h, w)
            )
            result2d = result2d.model_copy(
                update={
                    "tilt_axis_angle_deg": axis_deg,
                    "tilt_magnitude_deg": tilt_deg,
                }
            )
        # Copy phase-shift result from 1x1 fit into result2d when optimising phase
        if (
            fitting_params.optimize_phase_shift
            and result_1x1.phase_shift_degrees is not None
        ):
            result2d = result2d.model_copy(
                update={
                    "phase_shift_degrees": result_1x1.phase_shift_degrees,
                    "phase_shift_model_type": result_1x1.phase_shift_model_type,
                    "phase_shift_model": result_1x1.phase_shift_model,
                    "phase_shift_trace": result_1x1.phase_shift_trace,
                }
            )
        if results_path is not None:
            write_results_json(result2d, results_path)
        return mean_ps, result1d, result2d

    # -------------------------------------------------------------------------
    # Branch B: Single 2D defocus fit (grid or linear) over all patches
    # -------------------------------------------------------------------------
    # Optionally run 1x1 fit first for defocus_0 / get astig and phase to initialize
    fix_defocus_0_val = None
    initial_astigmatism_2d = 0.0
    initial_astigmatism_angle_2d = 0.0
    initial_phase_shift_for_2d = initial_phase_shift_2d
    if (
        fitting_params.defocus_model == "linear"
        and fitting_params.linear_fix_defocus_0_from_1x1
    ):
        result_1x1 = _estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=fitting_params.optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            n_iterations=fitting_params.n_iterations_2d,
            debug=fitting_params.debug,
            optimize_phase_shift=fitting_params.optimize_phase_shift,
            phase_shift_model=fitting_params.phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=fitting_params.phase_shift_quadratic_perpendicular_axis,
            initial_phase_shift=initial_phase_shift_2d,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
        )
        fix_defocus_0_val = float(result_1x1.defocus_model.data.mean().cpu().item())
        if result_1x1.astigmatism is not None:
            initial_astigmatism_2d = result_1x1.astigmatism
        if result_1x1.astigmatism_angle is not None:
            initial_astigmatism_angle_2d = result_1x1.astigmatism_angle
        if (
            fitting_params.optimize_phase_shift
            and result_1x1.phase_shift_degrees is not None
        ):
            initial_phase_shift_for_2d = result_1x1.phase_shift_degrees

    # Full 2D defocus optimisation (grid or linear model)
    result2d = estimate_ctf_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=normalised_patch_positions,
        defocus_grid_resolution=fitting_params.defocus_grid_resolution,
        frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
        initial_defocus=initial_defocus_2d,
        pixel_spacing_angstroms=new_spacing,
        debug=fitting_params.debug,
        optimize_astigmatism=fitting_params.optimize_astigmatism,
        defocus_model=fitting_params.defocus_model,
        initial_envelope_B=initial_envelope_B_2d,
        initial_astigmatism=initial_astigmatism_2d,
        initial_astigmatism_angle=initial_astigmatism_angle_2d,
        fix_defocus_0=fix_defocus_0_val,
        n_iterations=fitting_params.n_iterations_2d,
        optimize_phase_shift=fitting_params.optimize_phase_shift,
        phase_shift_model=fitting_params.phase_shift_model,
        phase_shift_quadratic_perpendicular_axis=fitting_params.phase_shift_quadratic_perpendicular_axis,
        initial_phase_shift=initial_phase_shift_for_2d,
        voltage_kev=optical_params.voltage_kev,
        spherical_aberration_mm=optical_params.spherical_aberration_mm,
        amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
        laser_params=laser_params,
    )
    # For linear defocus: add tilt axis and magnitude (degrees) to result
    if result2d.defocus_model_type == "linear":
        axis_deg, tilt_deg = linear_tilt_axis_and_magnitude_deg(
            result2d, new_spacing, min(h, w)
        )
        result2d = result2d.model_copy(
            update={
                "tilt_axis_angle_deg": axis_deg,
                "tilt_magnitude_deg": tilt_deg,
            }
        )
    # Optionally write defocus, phase shift, B envelope to JSON
    if results_path is not None:
        write_results_json(result2d, results_path)
    return mean_ps, result1d, result2d
