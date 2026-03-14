"""Estimate CTF from a 2D or 3D image."""

import math as _math
import warnings
from typing import Any, Literal, Optional

import einops
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_rescale import fourier_rescale_2d
from torch_grid_utils.patch_grid import patch_grid

from torch_ctf_estimation.estimate_defocus_1d import (
    estimate_defocus_1d,
    fit_background_spline_1d,
)
from torch_ctf_estimation.estimate_defocus_2d import (
    Defocus2DResults,
    estimate_defocus_2d,
    linear_tilt_axis_and_magnitude_deg,
)
from torch_ctf_estimation.models import LinearDefocusModel
from torch_ctf_estimation.utils.estimate_background_2d import estimate_background_2d
from torch_ctf_estimation.utils.normalize import normalize_image


def _estimate_defocus_2d_at_1x1(
    patch_power_spectra: torch.Tensor,
    defocus_grid_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus: float,
    pixel_spacing_angstroms: float,
    optimize_astigmatism: bool = False,
    initial_envelope_B: float = 0.0,
    debug: bool = False,
) -> Defocus2DResults:
    """
    Run 2D defocus estimation at 1x1 spatial resolution (center only).

    Averages patch power spectra over the spatial grid (gh, gw), builds
    single center positions per frame, and calls estimate_defocus_2d with
    defocus_grid_resolution=(nt, 1, 1) and grid model to get astigmatism
    and center defocus.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Shape (t, gh, gw, ph, pw).
    defocus_grid_resolution : tuple[int, int, int]
        (nt, nh, nw); nt is used for the time dimension.
    frequency_fit_range_angstroms, initial_defocus, pixel_spacing_angstroms :
        Passed through to estimate_defocus_2d.
    optimize_astigmatism : bool
        Whether to optimize astigmatism in the 2D fit.
    initial_envelope_B : float
        Initial B-factor for envelope.
    debug : bool
        If True, return debug info from 2D fit.

    Returns
    -------
    Defocus2DResults
        Result from 2D fit at 1x1 (defocus, astigmatism, envelope_B, etc.).
    """
    t, _gh, _gw, _ph, _pw = patch_power_spectra.shape
    nt = defocus_grid_resolution[0]
    device = patch_power_spectra.device
    # Mean over spatial patch grid -> (t, ph, pw)
    patch_ps_mean = patch_power_spectra.mean(dim=(1, 2))
    # (t, 1, 1, ph, pw)
    patch_ps_1x1 = patch_ps_mean.unsqueeze(1).unsqueeze(1)
    # Positions: (t, 1, 1, 3) with [t_norm, 0.5, 0.5]
    if t == 1:
        t_vals = torch.tensor([0.5], device=device, dtype=patch_power_spectra.dtype)
    else:
        t_vals = torch.linspace(0, 1, t, device=device, dtype=patch_power_spectra.dtype)
    positions_1x1 = torch.zeros(
        t, 1, 1, 3, device=device, dtype=patch_power_spectra.dtype
    )
    positions_1x1[:, 0, 0, 0] = t_vals
    positions_1x1[:, 0, 0, 1] = 0.5
    positions_1x1[:, 0, 0, 2] = 0.5
    return estimate_defocus_2d(
        patch_power_spectra=patch_ps_1x1,
        normalised_patch_positions=positions_1x1,
        defocus_grid_resolution=(nt, 1, 1),
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=initial_defocus,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=0.0,
        initial_astigmatism_angle=0.0,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
        defocus_model="grid",
        debug=debug,
    )


def _defocus_field_from_1d_fits(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    result_1x1: Defocus2DResults,
    defocus_model: Literal["grid", "linear"],
    defocus_grid_resolution: tuple[int, int, int],
    initial_defocus: float,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    defocus_range_microns: tuple[float, float],
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast_fraction: float,
    pixel_spacing_angstroms: float,
    optimize_envelope_1d: bool,
    b_range_1d: tuple[float, float],
    b_step_1d: float,
    refine_steps_1d: int,
    background_result: Optional[Any],
    device: torch.device,
) -> Defocus2DResults:
    """
    Build defocus field from per-patch 1D fits; fit grid or linear to those values.

    Runs estimate_defocus_1d on each patch, then fits either a 3D spline grid or
    a linear (defocus_0 + gradient) model to the (position, defocus_1d) data.
    Astigmatism and envelope come from result_1x1.
    """
    t, gh, gw, _ph, _pw = patch_power_spectra.shape
    nt, nh, nw = defocus_grid_resolution
    defocus_2d_center = float(result_1x1.defocus_model.data.mean().cpu().item())
    defocus_list = []
    for ti in range(t):
        for gi in range(gh):
            for gj in range(gw):
                ps = patch_power_spectra[ti, gi, gj]
                r1d = estimate_defocus_1d(
                    power_spectrum=ps,
                    image_sidelength=image_sidelength,
                    frequency_fit_range_angstroms=frequency_fit_range_angstroms,
                    defocus_range_microns=defocus_range_microns,
                    voltage_kev=voltage_kev,
                    spherical_aberration_mm=spherical_aberration_mm,
                    amplitude_contrast=amplitude_contrast_fraction,
                    pixel_spacing_angstroms=pixel_spacing_angstroms,
                    optimize_envelope=optimize_envelope_1d,
                    b_range=b_range_1d,
                    b_step=b_step_1d,
                    refine_steps=refine_steps_1d,
                    initial_defocus=defocus_2d_center,
                    background_result=background_result,
                )
                d = r1d.ctf_model.defocus_um
                if isinstance(d, torch.Tensor):
                    d = float(d.cpu().item())
                else:
                    d = float(d)
                defocus_list.append(d)
    defocus_vals = torch.tensor(
        defocus_list, device=device, dtype=patch_power_spectra.dtype
    ).view(t, gh, gw)
    positions_flat = normalised_patch_positions.reshape(-1, 3)
    defocus_flat = defocus_vals.reshape(-1, 1)

    astig = result_1x1.astigmatism or 0.0
    env_b = result_1x1.envelope_B
    envelope_B = float(env_b) if env_b is not None else None

    if defocus_model == "linear":
        defocus_0 = float(result_1x1.defocus_model.data.mean().cpu().item())
        design = torch.stack(
            [
                positions_flat[:, 1] - 0.5,
                positions_flat[:, 2] - 0.5,
            ],
            dim=1,
        )
        target = (defocus_flat.squeeze(1) - defocus_0).to(torch.float64).unsqueeze(1)
        design = design.to(torch.float64)
        sol = (torch.linalg.pinv(design) @ target).squeeze(1)
        # Replace NaN from singular/rank-deficient design with 0 for debuggable result
        sol = torch.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)
        u = float(sol[0].item())
        v = float(sol[1].item()) if sol.numel() > 1 else 0.0
        if _math.isnan(u):
            u = 0.0
        if _math.isnan(v):
            v = 0.0
        grad_mag = _math.sqrt(u * u + v * v)
        if _math.isnan(grad_mag) or grad_mag <= 0.0:
            grad_mag = 0.0
            angle_deg = 0.0
            warnings.warn(
                "Linear defocus gradient from 1D fits is zero or NaN (e.g. singular "
                "design matrix or no spatial defocus variation). Check patch positions "
                "and per-patch defocus values.",
                UserWarning,
                stacklevel=2,
            )
        else:
            angle_rad = _math.atan2(v, u)
            angle_deg = (angle_rad * 180.0 / _math.pi + 180.0) % 180.0
            if _math.isnan(angle_deg):
                angle_deg = 0.0
                warnings.warn(
                    "Linear defocus gradient angle was NaN; set to 0.",
                    UserWarning,
                    stacklevel=2,
                )
        linear_model = LinearDefocusModel(
            defocus_0=defocus_0,
            defocus_gradient_magnitude=grad_mag,
            defocus_gradient_angle=angle_deg,
        )
        mean_defocus = defocus_0
        return Defocus2DResults(
            defocus_model_type="linear",
            defocus_model=linear_model,
            astigmatism=astig,
            astigmatism_angle=result_1x1.astigmatism_angle or 0.0,
            envelope_B=envelope_B,
            defocus_u=mean_defocus + astig / 2.0,
            defocus_v=mean_defocus - astig / 2.0,
        )
    # grid: fit 3D spline to (positions, defocus_flat)
    grid_data = (
        torch.ones((nt, nh, nw), device=device, dtype=patch_power_spectra.dtype)
        * initial_defocus
    )
    grid_model = CubicCatmullRomGrid3d.from_grid_data(grid_data).to(device)
    optimiser = torch.optim.Adam(grid_model.parameters(), lr=0.01)
    n_fit_steps = 100
    for _ in range(n_fit_steps):
        optimiser.zero_grad()
        pred = grid_model(positions_flat).squeeze(-1)
        loss = ((pred - defocus_flat.squeeze(1)) ** 2).mean()
        loss.backward()
        optimiser.step()
    mean_defocus = float(grid_model.data.mean().cpu().item())
    return Defocus2DResults(
        defocus_model_type="grid",
        defocus_model=grid_model,
        astigmatism=astig,
        astigmatism_angle=result_1x1.astigmatism_angle or 0.0,
        envelope_B=envelope_B,
        defocus_u=mean_defocus + astig / 2.0,
        defocus_v=mean_defocus - astig / 2.0,
    )


def estimate_ctf(
    image: torch.Tensor,  # (t, h, w) or (h, w)
    pixel_spacing_angstroms: float,
    defocus_grid_resolution: tuple[int, int, int],  # (t, h, w); linear uses nt only
    frequency_fit_range_angstroms: tuple[float, float],  # (low, high)
    defocus_range_microns: tuple[float, float],  # (low, high)
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast_fraction: float,
    patch_sidelength: int = 256,
    device: torch.device = None,
    debug: bool = False,
    optimize_astigmatism: bool = False,
    defocus_model: Literal["grid", "linear"] = "grid",
    use_1d_defocus_for_spatial: bool = False,
    linear_fix_defocus_0_from_1x1: bool = False,
    refine_steps_1d: int = 40,
    optimize_envelope_1d: bool = True,
    b_range_1d: tuple[float, float] = (0.0, 100.0),
    b_step_1d: float = 1.0,
    initial_envelope_B: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Estimate CTF from a 2D or 3D image.

    Parameters
    ----------
    image: torch.Tensor
        §`(t, h, w)` or `(h, w)` array containing 2D or 3D image data.
    pixel_spacing_angstroms: float
        Isotropic pixel spacing in angstroms.
    defocus_grid_resolution: tuple[int, int, int]
        Resolution of the defocus grid ``(nt, nh, nw)``. When ``defocus_model="linear"``
        only the first element (nt) is used; nh and nw are ignored.
    frequency_fit_range_angstroms: tuple[float, float]
        `(low, high)` spatial frequency cutoffs for fitting in angstroms.
    defocus_range_microns: tuple[float, float]
        `(low, high)` defoci in microns for initial 1D fit.
    voltage_kev: float
        Acceleration voltage in keV.
    spherical_aberration_mm: float
        Spherical aberration in mm.
    amplitude_contrast_fraction: float
        Amplitude contrast fraction.
    patch_sidelength: int
        If >= 0: sidelength of the patches to extract from the image (with 50% overlap).
        If < 0: whole-image mode—no patching; the full image is used as a single patch
        per frame, and defocus_grid_resolution must have nh=1 and nw=1.
    device: torch.device, optional
        Device for computation. If None, uses ``cuda:0`` when available, else ``cpu``.
    debug: bool
        Whether to return debug information.
    optimize_astigmatism: bool
        Whether to optimize the astigmatism.
    defocus_model: {"grid", "linear"}, optional
        Defocus model: "grid" (3D spline over t,x,y) or "linear" (tilt model).
        Default "grid".
    use_1d_defocus_for_spatial: bool, optional
        When True: get astigmatism from 2D fit at 1x1, then build defocus field
        from per-patch 1D gradient-descent refinement only (no 1D grid search),
        using the 2D defocus as initial for each patch. Fit grid or linear to
        those refined values. For grid model this runs only when (nh, nw) > (1, 1);
        for linear model it can run at any resolution (including 1x1). Default False.
    linear_fix_defocus_0_from_1x1: bool, optional
        When True and defocus_model is "linear": run 2D fit at 1x1 first, fix
        defocus_0 to that value, and fit only gradient (and t-splines if nt>1)
        via 2D ZNCC. For linear model with gradient from 1D fits, use
        use_1d_defocus_for_spatial=True instead. Default False.
    refine_steps_1d: int, optional
        Number of gradient-descent refinement steps for 1D defocus (mean and
        per-patch when use_1d_defocus_for_spatial). Default 40.
    optimize_envelope_1d: bool
        Whether to optimize the envelope in 1D.
    b_range_1d: tuple[float, float]
        `(low, high)` B-factor range for envelope optimization in 1D.
    b_step_1d: float
        Step size for envelope optimization in 1D.
    initial_envelope_B: Optional[float]
        Initial B-factor for envelope optimization in 2D.

    Returns
    -------
    mean_ps: torch.Tensor
        Mean power spectrum of the patches.
    result1d: Defocus1DResults
        Results from 1D defocus estimation.
    result2d: Defocus2DResults
        Results from 2D defocus estimation.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # coerce to float and move to device
    image = image.float().to(device)

    # pack image to (t, h, w)
    image, _ = einops.pack([image], pattern="* h w")

    # normalize images to mean 0 std 1
    image = normalize_image(image)
    # cuton, cutoff = frequency_fit_range_angstroms
    # target_spacing = 0.5 * cutoff
    new_spacing = max(3.0, pixel_spacing_angstroms)
    image, _ = fourier_rescale_2d(
        image=image, source_spacing=pixel_spacing_angstroms, target_spacing=new_spacing
    )
    t, h, w = image.shape

    use_whole_image = patch_sidelength < 0
    if use_whole_image:
        print("Using whole-image mode")
        nt, nh, nw = defocus_grid_resolution
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
            patch_shape=(1, patch_sidelength, patch_sidelength),
            patch_step=(1, patch_sidelength // 2, patch_sidelength // 2),
        )
        image_sidelength_for_1d = patch_sidelength

    patches = einops.rearrange(patches, "t gh gw 1 ph pw -> t gh gw ph pw")

    # calculate power spectra of all patches and mean of all ps
    patch_ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2
    mean_ps = einops.reduce(patch_ps, "... ph pw -> ph pw", reduction="mean")

    nt, nh, nw = defocus_grid_resolution
    use_1d_spatial = use_1d_defocus_for_spatial and (
        defocus_model == "linear" or (nh > 1 or nw > 1)
    )
    # When using per-patch 1D defocus, fit background once on mean and reuse for all
    bg_mean: Optional[Any] = None
    if use_1d_spatial:
        bg_mean = fit_background_spline_1d(
            power_spectrum=mean_ps,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=frequency_fit_range_angstroms,
            pixel_spacing_angstroms=new_spacing,
        )

    # estimate defocus in 1D from mean of power spectra
    result1d = estimate_defocus_1d(
        power_spectrum=mean_ps,
        image_sidelength=image_sidelength_for_1d,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast_fraction,
        pixel_spacing_angstroms=new_spacing,
        optimize_envelope=optimize_envelope_1d,
        b_range=b_range_1d,
        b_step=b_step_1d,
        refine_steps=refine_steps_1d,
        background_result=bg_mean,
    )

    # estimate 2D background and subtract prior to 2D defocus estimation
    background_2d = estimate_background_2d(
        power_spectrum=mean_ps,
        image_sidelength=image_sidelength_for_1d,
    )
    patch_ps -= background_2d

    # estimate defocus in 2D with gradient based optimisation
    image_dimension_lengths = (
        torch.tensor([t - 1, h - 1, w - 1]).float().to(patch_ps.device)
    )
    normalised_patch_positions = patch_centers / image_dimension_lengths

    initial_envelope_B_2d = initial_envelope_B
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

    if use_1d_spatial:
        result_1x1 = _estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=defocus_grid_resolution,
            frequency_fit_range_angstroms=frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            debug=debug,
        )
        result2d = _defocus_field_from_1d_fits(
            patch_power_spectra=patch_ps,
            normalised_patch_positions=normalised_patch_positions,
            result_1x1=result_1x1,
            defocus_model=defocus_model,
            defocus_grid_resolution=defocus_grid_resolution,
            initial_defocus=initial_defocus_2d,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=frequency_fit_range_angstroms,
            defocus_range_microns=defocus_range_microns,
            voltage_kev=voltage_kev,
            spherical_aberration_mm=spherical_aberration_mm,
            amplitude_contrast_fraction=amplitude_contrast_fraction,
            pixel_spacing_angstroms=new_spacing,
            optimize_envelope_1d=optimize_envelope_1d,
            b_range_1d=b_range_1d,
            b_step_1d=b_step_1d,
            refine_steps_1d=refine_steps_1d,
            background_result=bg_mean,
            device=patch_ps.device,
        )
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
        return mean_ps, result1d, result2d

    fix_defocus_0_val = None
    initial_astigmatism_2d = 0.0
    initial_astigmatism_angle_2d = 0.0
    if defocus_model == "linear" and linear_fix_defocus_0_from_1x1:
        result_1x1 = _estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=defocus_grid_resolution,
            frequency_fit_range_angstroms=frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            debug=debug,
        )
        fix_defocus_0_val = float(result_1x1.defocus_model.data.mean().cpu().item())
        if result_1x1.astigmatism is not None:
            initial_astigmatism_2d = result_1x1.astigmatism
        if result_1x1.astigmatism_angle is not None:
            initial_astigmatism_angle_2d = result_1x1.astigmatism_angle

    result2d = estimate_defocus_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=normalised_patch_positions,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=initial_defocus_2d,
        pixel_spacing_angstroms=new_spacing,
        debug=debug,
        optimize_astigmatism=optimize_astigmatism,
        defocus_model=defocus_model,
        initial_envelope_B=initial_envelope_B_2d,
        initial_astigmatism=initial_astigmatism_2d,
        initial_astigmatism_angle=initial_astigmatism_angle_2d,
        fix_defocus_0=fix_defocus_0_val,
    )
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
    return mean_ps, result1d, result2d
