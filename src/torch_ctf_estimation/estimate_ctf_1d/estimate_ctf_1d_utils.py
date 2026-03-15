"""Utility functions for 1D CTF estimation (background fit, grid search, refinement)."""

import math
from typing import Optional

import einops
import torch
from torch_ctf import calculate_ctf_1d
from torch_cubic_spline_grids import CubicBSplineGrid1d
from torch_fourier_filter.dft_utils import rotational_average_dft_2d
from torch_grid_utils.fftfreq_grid import (
    fftfreq_to_spatial_frequency,
    spatial_frequency_to_fftfreq,
)

from torch_ctf_estimation.models.results_models import (
    _Background1DResult,
    _GridSearch1DResult,
)


def get_background_result(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    pixel_spacing_angstroms: float,
    background_result: _Background1DResult | None = None,
) -> _Background1DResult:
    """
    Get background-subtracted 1D spectrum: reuse pre-fitted or fit new spline.

    If background_result is provided, rotationally average this spectrum and
    subtract the pre-fitted background. Otherwise fit a cubic B-spline to the
    rotationally averaged power spectrum and return the result.

    Parameters
    ----------
    power_spectrum : torch.Tensor
        (h, w) array containing 2D rfft (no fftshift applied).
    image_sidelength : int
        Sidelength of 2D images prior to rfft calculation.
    frequency_fit_range_angstroms : tuple[float, float]
        (low, high) spatial frequency cutoffs for fitting in angstroms.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    background_result : _Background1DResult, optional
        If provided, reuse this pre-fitted background instead of fitting.

    Returns
    -------
    _Background1DResult
        Background model and background-subtracted raps in fit range.
    """
    if background_result is not None:
        # Reuse path: rotationally average this spectrum, subtract pre-fitted background
        device = power_spectrum.device
        h, w = image_sidelength, image_sidelength
        ps_cpu = power_spectrum.cpu()
        rotationally_averaged_power_spectrum, _ = rotational_average_dft_2d(
            ps_cpu,
            image_shape=(h, w),
            rfft=True,
            fftshifted=False,
        )
        rotationally_averaged_power_spectrum = rotationally_averaged_power_spectrum.to(
            device
        )
        fit_mask = background_result.fit_mask
        raps_in_fit_range = rotationally_averaged_power_spectrum[fit_mask].clone()
        x = torch.linspace(
            0,
            1,
            steps=len(raps_in_fit_range),
            device=device,
            dtype=raps_in_fit_range.dtype,
        )
        background = torch.exp(
            background_result.background_model(x).squeeze().to(device)
        )
        raps_in_fit_range = (raps_in_fit_range - background).detach()
        return _Background1DResult(
            rotationally_averaged_power_spectrum=rotationally_averaged_power_spectrum,
            freqs=background_result.freqs.to(device),
            spatial_freqs=background_result.spatial_freqs.to(device),
            fit_mask=fit_mask.to(device),
            raps_in_fit_range=raps_in_fit_range,
            background_model=background_result.background_model,
        )
    # Fit path: fit new B-spline to rotationally averaged spectrum
    return fit_background_spline_1d(
        power_spectrum=power_spectrum,
        image_sidelength=image_sidelength,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
    )


def fit_background_spline_1d(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    pixel_spacing_angstroms: float,
    n_spline_iterations: int = 200,
) -> _Background1DResult:
    """
    Fit a cubic B-spline background to the 1D rotationally averaged power spectrum.

    Returns background-subtracted values in the fit range.

    Parameters
    ----------
    power_spectrum : torch.Tensor
        (h, w) array containing 2D rfft (no fftshift applied).
    image_sidelength : int
        Sidelength of 2D images prior to rfft calculation.
    frequency_fit_range_angstroms : tuple[float, float]
        (low, high) spatial frequency cutoffs for fitting in angstroms.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    n_spline_iterations : int
        Number of Adam steps for spline fitting. Default 200.

    Returns
    -------
    _Background1DResult
        Background model, background-subtracted raps in fit range, freqs, spatial_freqs,
        fit_mask, and full rotationally averaged power spectrum.
    """
    # Rotationally average 2D power spectrum to 1D (dft_utils uses CPU; move back after)
    device = power_spectrum.device
    h, w = image_sidelength, image_sidelength
    ps_cpu = power_spectrum.cpu()
    rotationally_averaged_power_spectrum, _ = rotational_average_dft_2d(
        ps_cpu,
        image_shape=(h, w),
        rfft=True,
        fftshifted=False,
    )
    rotationally_averaged_power_spectrum = rotationally_averaged_power_spectrum.to(
        device
    )

    # Build frequency grid and mask for fit range (angstroms -> fftfreq)
    freqs = torch.fft.rfftfreq(h, device=device)
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(
        1 / low_ang, spacing=pixel_spacing_angstroms
    )
    high_fftfreq = spatial_frequency_to_fftfreq(
        1 / high_ang, spacing=pixel_spacing_angstroms
    )
    fit_mask = torch.logical_and(freqs >= low_fftfreq, freqs <= high_fftfreq)
    raps_in_fit_range = rotationally_averaged_power_spectrum[fit_mask].clone()

    spatial_freqs = fftfreq_to_spatial_frequency(freqs, pixel_spacing_angstroms)

    # Fit B-spline in log(raps); subtract exp(spline) for background-subtracted spectrum
    background_model = CubicBSplineGrid1d(resolution=3).to(device)
    background_optimiser = torch.optim.Adam(params=background_model.parameters(), lr=1)
    x = torch.linspace(0, 1, steps=len(raps_in_fit_range), device=device)
    y = torch.log(raps_in_fit_range)

    # Adam on MSE(log(raps), spline(x)) so spline approximates log background
    for _ in range(n_spline_iterations):
        prediction = background_model(x).squeeze()
        difference = prediction - y
        mean_squared_error = torch.mean(difference**2)
        mean_squared_error.backward()
        background_optimiser.step()
        background_optimiser.zero_grad()

    # Subtract exp(spline) from raps to get background-subtracted spectrum in fit range
    background = torch.exp(background_model(x).squeeze())
    raps_in_fit_range = (raps_in_fit_range - background).detach()

    return _Background1DResult(
        rotationally_averaged_power_spectrum=rotationally_averaged_power_spectrum,
        freqs=freqs,
        spatial_freqs=spatial_freqs,
        fit_mask=fit_mask,
        raps_in_fit_range=raps_in_fit_range,
        background_model=background_model,
    )


def _grid_search_best_from_zncc(
    zncc: torch.Tensor,
    test_defoci: torch.Tensor,
    test_B_values: torch.Tensor | None,
    test_phase_shift_values: torch.Tensor | None,
    optimize_envelope: bool,
    optimize_phase_shift: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    From ZNCC tensor(s), compute best (defocus, B, phase) and correlation curves.

    Four modes: (envelope, phase) = (T,T), (T,F), (F,T), (F,F). ZNCC is either
    flat (envelope cases) or 2D/1D (no-envelope). Returns best_defocus, best_B,
    best_phase_shift, cross_correlations_1d, cross_correlations_2d.
    """
    if optimize_envelope and optimize_phase_shift:
        assert test_B_values is not None and test_phase_shift_values is not None
        n_p = test_phase_shift_values.shape[0]
        n_defocus = test_defoci.shape[0]
        n_B = test_B_values.shape[0]
        zncc_3d = zncc.reshape(n_p, n_defocus, n_B)
        cross_correlations_2d = zncc_3d.max(dim=0).values
        cross_correlations_1d = torch.amax(zncc_3d, dim=(0, 2))
        idx = torch.argmax(zncc).item()
        best_p_idx = idx // (n_defocus * n_B)
        rest = idx % (n_defocus * n_B)
        best_defocus_idx = rest // n_B
        best_B_idx = rest % n_B
        best_defocus = test_defoci[best_defocus_idx]
        best_B = test_B_values[best_B_idx]
        best_phase_shift = test_phase_shift_values[best_p_idx]
    elif optimize_envelope and not optimize_phase_shift:
        assert test_B_values is not None
        n_defocus = test_defoci.shape[0]
        n_B = test_B_values.shape[0]
        zncc_2d = zncc.reshape(n_defocus, n_B)
        cross_correlations_1d = zncc_2d.max(dim=1).values
        cross_correlations_2d = zncc_2d
        max_correlation_idx = torch.argmax(zncc)
        best_defocus_idx = max_correlation_idx // n_B
        best_B_idx = max_correlation_idx % n_B
        best_defocus = test_defoci[best_defocus_idx]
        best_B = test_B_values[best_B_idx]
        best_phase_shift = None
    elif not optimize_envelope and optimize_phase_shift:
        assert test_phase_shift_values is not None
        cross_correlations_1d = zncc.max(dim=0).values
        max_correlation_idx = torch.argmax(zncc)
        best_p_idx = (max_correlation_idx // zncc.shape[1]).item()
        best_d_idx = (max_correlation_idx % zncc.shape[1]).item()
        best_defocus = test_defoci[best_d_idx]
        best_B = None
        best_phase_shift = test_phase_shift_values[best_p_idx]
        cross_correlations_2d = None
    else:
        cross_correlations_1d = zncc
        cross_correlations_2d = None
        best_defocus = test_defoci[torch.argmax(zncc)]
        best_B = None
        best_phase_shift = None
    return (
        best_defocus,
        best_B,
        best_phase_shift,
        cross_correlations_1d,
        cross_correlations_2d,
    )


def grid_search_defocus_and_envelope_1d(
    raps_in_fit_range: torch.Tensor,
    spatial_freqs: torch.Tensor,
    fit_mask: torch.Tensor,
    image_sidelength: int,
    defocus_range_microns: tuple[float, float],
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    optimize_envelope: bool = True,
    b_range: tuple[float, float] = (0.0, 100.0),
    b_step: float = 1.0,
    defocus_step: float = 0.01,
    optimize_phase_shift: bool = False,
    phase_shift_step: float = 5.0,
) -> _GridSearch1DResult:
    """
    Grid search over defocus (and optionally B-factor envelope) to maximise ZNCC.

    Uses the background-subtracted 1D power spectrum in the fit range.

    Parameters
    ----------
    raps_in_fit_range : torch.Tensor
        Background-subtracted rotationally averaged power spectrum in the fit range.
    spatial_freqs : torch.Tensor
        Spatial frequencies (1/Å) for the full 1D spectrum.
    fit_mask : torch.Tensor
        Boolean mask indicating which frequencies are in the fit range.
    image_sidelength : int
        Sidelength of 2D images (used for CTF n_samples).
    defocus_range_microns : tuple[float, float]
        (low, high) defoci in microns for the grid.
    voltage_kev, spherical_aberration_mm, amplitude_contrast : float
        CTF parameters.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    optimize_envelope : bool
        If True, also grid over B-factor envelope.
    b_range : tuple[float, float]
        (low, high) B-factor range when optimize_envelope is True.
    b_step : float
        B-factor grid step when optimize_envelope is True.
    defocus_step : float
        Defocus grid step in microns.
    optimize_phase_shift : bool
        If True, also grid over phase shift (0-180° in steps of phase_shift_step).
    phase_shift_step : float
        Phase shift grid step in degrees when optimize_phase_shift is True. Default 5.0.

    Returns
    -------
    _GridSearch1DResult
        Best defocus, best B (or None), test defoci, cross correlations, test B values,
        and optionally best_phase_shift and test_phase_shift_values.
    """
    device = raps_in_fit_range.device
    dtype = raps_in_fit_range.dtype
    h = image_sidelength
    # Normalise observed spectrum so ZNCC is just inner product with normalised model
    normalised_raps_in_fit_range = raps_in_fit_range / torch.linalg.norm(
        raps_in_fit_range
    )

    # Build defocus grid (and optional phase-shift grid)
    test_defoci = torch.arange(
        start=defocus_range_microns[0],
        end=defocus_range_microns[1] + defocus_step,
        step=defocus_step,
        device=device,
        dtype=dtype,
    )
    test_phase_shift_values = None
    if optimize_phase_shift:
        test_phase_shift_values = torch.arange(
            0, 180, phase_shift_step, device=device, dtype=dtype
        )

    # Simulate CTF² for all test defoci (and optionally each phase); slice to fit range
    if not optimize_phase_shift:
        ctf2 = (
            calculate_ctf_1d(
                defocus=test_defoci,
                voltage=voltage_kev,
                spherical_aberration=spherical_aberration_mm,
                amplitude_contrast=amplitude_contrast,
                phase_shift=0,
                pixel_size=pixel_spacing_angstroms,
                n_samples=h // 2 + 1,
                oversampling_factor=3,
            )
            ** 2
        )
        simulated_ctf2_in_fit_range = ctf2[:, fit_mask]
    else:
        # Grid over phase: one CTF^2 grid per phase, stack (n_phase, n_defocus, n_freq)
        assert test_phase_shift_values is not None  # set when optimize_phase_shift
        n_p = test_phase_shift_values.shape[0]
        list_ctf2 = []
        for i in range(n_p):
            p = test_phase_shift_values[i].item()
            ctf2_p = (
                calculate_ctf_1d(
                    defocus=test_defoci,
                    voltage=voltage_kev,
                    spherical_aberration=spherical_aberration_mm,
                    amplitude_contrast=amplitude_contrast,
                    phase_shift=p,
                    pixel_size=pixel_spacing_angstroms,
                    n_samples=h // 2 + 1,
                    oversampling_factor=3,
                )
                ** 2
            )
            list_ctf2.append(ctf2_p)
        simulated_ctf2_in_fit_range = torch.stack(list_ctf2, dim=0)[:, :, fit_mask]

    # ZNCC vs normalised model; unpack best (defocus, B, phase) via helper
    if optimize_envelope:
        b_low, b_high = b_range
        if b_step <= 0:
            raise ValueError("b_step must be positive.")
        test_B_values = torch.arange(
            start=b_low,
            end=b_high + b_step,
            step=b_step,
            device=device,
            dtype=dtype,
        )
        # B-factor envelope: exp(-B * s^2 / 2); apply to CTF² then normalise and ZNCC
        env_power_full = torch.exp(
            -(test_B_values[:, None] * spatial_freqs[None, :] ** 2) / 2.0
        )
        env_power_in_fit_range = env_power_full[:, fit_mask]
        if optimize_phase_shift:
            # simulated_ctf2_in_fit_range: (n_p, n_defocus, n_fit)
            simulated_ctf2_expanded = (
                simulated_ctf2_in_fit_range[:, :, None, :]
                * (env_power_in_fit_range[None, None, :, :])
            )
            n_fit = simulated_ctf2_expanded.shape[-1]
            simulated_ctf2_flat = simulated_ctf2_expanded.reshape(-1, n_fit)
        else:
            simulated_ctf2_expanded = simulated_ctf2_in_fit_range[:, None, :]
            simulated_ctf2_with_env = (
                simulated_ctf2_expanded * env_power_in_fit_range[None, :, :]
            )
            n_fit = simulated_ctf2_with_env.shape[-1]
            simulated_ctf2_flat = simulated_ctf2_with_env.reshape(-1, n_fit)
        simulated_ctf2_flat = simulated_ctf2_flat / torch.linalg.norm(
            simulated_ctf2_flat, dim=-1, keepdim=True
        )
        zncc = einops.einsum(
            simulated_ctf2_flat,
            normalised_raps_in_fit_range,
            "b i, i -> b",
        )
    else:
        test_B_values = None
        simulated_ctf2_in_fit_range = simulated_ctf2_in_fit_range / torch.linalg.norm(
            simulated_ctf2_in_fit_range, dim=-1, keepdim=True
        )
        if optimize_phase_shift:
            zncc = einops.einsum(
                simulated_ctf2_in_fit_range,
                normalised_raps_in_fit_range,
                "p d i, i -> p d",
            )
        else:
            zncc = einops.einsum(
                simulated_ctf2_in_fit_range,
                normalised_raps_in_fit_range,
                "b i, i -> b",
            )

    result = _grid_search_best_from_zncc(
        zncc,
        test_defoci,
        test_B_values,
        test_phase_shift_values,
        optimize_envelope,
        optimize_phase_shift,
    )
    (
        best_defocus,
        best_B,
        best_phase_shift,
        cross_correlations_1d,
        cross_correlations_2d,
    ) = result

    return _GridSearch1DResult(
        best_defocus=best_defocus,
        best_B=best_B,
        test_defoci=test_defoci,
        cross_correlations_1d=cross_correlations_1d,
        cross_correlations_2d=cross_correlations_2d,
        test_B_values=test_B_values,
        best_phase_shift=best_phase_shift,
        test_phase_shift_values=test_phase_shift_values,
    )


def refine_defocus_and_b_factor_1d(
    initial_defocus: float,
    initial_B: Optional[float],
    raps_in_fit_range: torch.Tensor,
    spatial_freqs: torch.Tensor,
    fit_mask: torch.Tensor,
    image_sidelength: int,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    defocus_range_microns: tuple[float, float],
    optimize_envelope: bool,
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    b_factor_lr: float = 1.0,
    initial_phase_shift: Optional[float] = None,
    optimize_phase_shift: bool = False,
    phase_shift_lr: float = 5.0,
    phase_shift_range: tuple[float, float] = (0.0, 180.0),
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Refine defocus (and optionally B factor) by gradient descent to maximise ZNCC.

    Uses the background-subtracted 1D power spectrum in the fit range.

    Parameters
    ----------
    initial_defocus : float
        Defocus in microns from grid search.
    initial_B : float or None
        B factor from grid search, or None if envelope was not optimised.
    raps_in_fit_range : torch.Tensor
        Background-subtracted rotationally averaged power spectrum in the fit range.
    spatial_freqs : torch.Tensor
        Spatial frequencies (1/Å) for the full 1D spectrum.
    fit_mask : torch.Tensor
        Boolean mask indicating which frequencies are in the fit range.
    image_sidelength : int
        Sidelength of 2D images (used for CTF n_samples).
    voltage_kev, spherical_aberration_mm, amplitude_contrast : float
        CTF parameters.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    defocus_range_microns : tuple[float, float]
        (low, high) defocus bounds; defocus is clamped to this range each step.
    optimize_envelope : bool
        If True, also refine B factor (initial_B must be provided).
    n_iterations : int
        Number of gradient descent steps. Default 100.
    defocus_lr : float
        Learning rate for defocus. Default 0.001.
    b_factor_lr : float
        Learning rate for B factor when optimize_envelope is True. Default 0.1.
    initial_phase_shift : float, optional
        Initial phase shift in degrees when optimize_phase_shift is True.
    optimize_phase_shift : bool
        If True, also refine phase shift (clamped to phase_shift_range each step).
    phase_shift_lr : float
        Learning rate for phase shift. Default 1.0.
    phase_shift_range : tuple[float, float]
        (low, high) phase shift bounds in degrees. Default (0.0, 180.0).

    Returns
    -------
    tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
        (refined_defocus, refined_B, refined_phase_shift).
        refined_phase_shift is None when optimize_phase_shift is False.
    """
    device = raps_in_fit_range.device
    dtype = raps_in_fit_range.dtype
    # Detach observation so gradients flow only through defocus/B/phase params
    raps_detached = raps_in_fit_range.detach()
    normalised_raps = raps_detached / torch.linalg.norm(raps_detached)
    spatial_freqs_fit = spatial_freqs[fit_mask].detach()
    defocus_low, defocus_high = defocus_range_microns
    phase_shift_low, phase_shift_high = phase_shift_range

    # Parameters: defocus (always), B and phase (u,v) if requested
    defocus_param = torch.nn.Parameter(
        torch.tensor(initial_defocus, device=device, dtype=dtype)
    )
    param_list: list[torch.nn.Parameter] = [defocus_param]
    if optimize_envelope and initial_B is not None:
        b_param = torch.nn.Parameter(
            torch.tensor(initial_B, device=device, dtype=dtype)
        )
        param_list.append(b_param)
    else:
        b_param = None
    if optimize_phase_shift and initial_phase_shift is not None:
        # (u,v) representation: u = cos(2*theta), v = sin(2*theta);
        # theta = 0.5*atan2(v,u), 0-180°
        theta_rad = initial_phase_shift * (math.pi / 180.0)
        u_param = torch.nn.Parameter(
            torch.tensor(math.cos(2.0 * theta_rad), device=device, dtype=dtype)
        )
        v_param = torch.nn.Parameter(
            torch.tensor(math.sin(2.0 * theta_rad), device=device, dtype=dtype)
        )
    else:
        u_param = v_param = None

    optimiser = torch.optim.Adam(
        [
            {"params": [defocus_param], "lr": defocus_lr},
            *(
                [{"params": [b_param], "lr": b_factor_lr}]
                if b_param is not None
                else []
            ),
            *(
                [{"params": [u_param, v_param], "lr": phase_shift_lr}]
                if u_param is not None and v_param is not None
                else []
            ),
        ]
    )

    PHASE_SHIFT_UNIT_CIRCLE_PENALTY = 0.1

    # Gradient steps: clamp defocus, get phase from (u,v), simulate CTF², maximise ZNCC
    h = image_sidelength
    for _ in range(n_iterations):
        optimiser.zero_grad()
        with torch.no_grad():
            defocus_param.clamp_(min=defocus_low, max=defocus_high)
        # Phase in degrees from unit-circle (u,v): theta = 0.5 * atan2(v, u)
        if u_param is not None and v_param is not None:
            phase_val = torch.remainder(
                0.5 * torch.atan2(v_param, u_param) * (180.0 / math.pi), 180.0
            )
        else:
            phase_val = 0.0
        ctf2 = (
            calculate_ctf_1d(
                defocus=defocus_param.unsqueeze(0),
                voltage=voltage_kev,
                spherical_aberration=spherical_aberration_mm,
                amplitude_contrast=amplitude_contrast,
                phase_shift=phase_val if (u_param is not None) else 0,
                pixel_size=pixel_spacing_angstroms,
                n_samples=h // 2 + 1,
                oversampling_factor=3,
            )
            ** 2
        )
        # Model = CTF^2 * B envelope (if used); ZNCC vs normalised data
        ctf2_fit = ctf2.squeeze(0)[fit_mask]
        if b_param is not None:
            envelope = torch.exp(-(b_param * spatial_freqs_fit**2) / 2.0)
            model_fit = ctf2_fit * envelope
        else:
            model_fit = ctf2_fit
        model_fit = model_fit / torch.linalg.norm(model_fit)
        zncc = einops.einsum(model_fit, normalised_raps, "i, i ->")
        loss = -zncc
        # Keep (u,v) on unit circle via penalty
        if u_param is not None and v_param is not None:
            penalty = (u_param**2 + v_param**2 - 1.0) ** 2
            loss = loss + PHASE_SHIFT_UNIT_CIRCLE_PENALTY * penalty
        loss.backward()
        optimiser.step()
        # Re-project (u,v) onto unit circle after step (clamp phase to range)
        if u_param is not None and v_param is not None:
            with torch.no_grad():
                phase_deg = (
                    0.5 * torch.atan2(v_param, u_param) * (180.0 / math.pi)
                ).item()
                phase_deg = max(phase_shift_low, min(phase_shift_high, phase_deg))
                theta_rad = phase_deg * (math.pi / 180.0)
                u_param.data.fill_(math.cos(2.0 * theta_rad))
                v_param.data.fill_(math.sin(2.0 * theta_rad))

    # Final clamp and extract (refined_defocus, refined_B, refined_phase_shift)
    with torch.no_grad():
        defocus_param.clamp_(min=defocus_low, max=defocus_high)
    refined_defocus = defocus_param.detach()
    refined_B = b_param.detach() if b_param is not None else None
    if u_param is not None and v_param is not None:
        raw_deg = 0.5 * math.degrees(
            math.atan2(v_param.detach().cpu().item(), u_param.detach().cpu().item())
        )
        refined_phase_shift = raw_deg % 180.0
        refined_phase_shift = min(refined_phase_shift, 180.0 - refined_phase_shift)
    else:
        refined_phase_shift = None
    return refined_defocus, refined_B, refined_phase_shift
