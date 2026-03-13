"""Estimate defocus in 1D from a power spectrum."""

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

from .models import CTF, Defocus1DResults

__all__ = ["Defocus1DResults", "estimate_defocus_1d"]


def estimate_defocus_1d(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    defocus_range_microns: tuple[float, float],
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    optimize_envelope: bool = True,
    b_range: tuple[float, float] = (0.0, 100.0),
    b_step: float = 1.0,
) -> Defocus1DResults:
    """
    Estimate defocus in 1D from a power spectrum.

    Parameters
    ----------
    power_spectrum: torch.Tensor
        `(h, w)` array containing 2D rfft (no fftshift applied).
    image_sidelength: int
        Sidelength of 2D images prior to rfft calculation.
    frequency_fit_range_angstroms: tuple[float, float]
        `(low, high)` spatial frequency cutoffs for fitting in angstroms.
    defocus_range_microns: tuple[float, float]
        `(low, high)` defoci in microns for initial 1D fit.
    voltage_kev: float
        Acceleration voltage in keV.
    spherical_aberration_mm: float
        Spherical aberration in mm.
    amplitude_contrast: float
        Amplitude contrast fraction.
    pixel_spacing_angstroms: float
        Isotropic pixel spacing in angstroms.
    optimize_envelope: bool
        Whether to optimize the envelope.
    b_range: tuple[float, float]
        `(low, high)` B-factor range for envelope optimization.
    b_step: float
        Step size for envelope optimization.

    Returns
    -------
    Defocus1DResults
        Results from 1D defocus estimation containing frequencies, power spectrum,
        background model, and CTF fitting results.
    """
    # calculate 1d rotationally averaged power spectrum
    h, w = image_sidelength, image_sidelength
    rotationally_averaged_power_spectrum, _ = rotational_average_dft_2d(
        power_spectrum,
        image_shape=(h, w),
        rfft=True,
        fftshifted=False,
    )

    # determine subset of 1D values to use based on fit range
    freqs = torch.fft.rfftfreq(h)
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(
        1 / low_ang, spacing=pixel_spacing_angstroms
    )
    high_fftfreq = spatial_frequency_to_fftfreq(
        1 / high_ang, spacing=pixel_spacing_angstroms
    )
    fit_mask = torch.logical_and(freqs >= low_fftfreq, freqs <= high_fftfreq)
    raps_in_fit_range = rotationally_averaged_power_spectrum[fit_mask]

    # corresponding spatial frequencies in 1/Å for envelope modelling
    spatial_freqs = fftfreq_to_spatial_frequency(freqs, pixel_spacing_angstroms)

    # estimate 1D background by fitting a cubic B-spline with 3 control points
    # fit to log(values) for numerical stability
    background_model = CubicBSplineGrid1d(resolution=3)
    background_optimiser = torch.optim.Adam(params=background_model.parameters(), lr=1)
    x = torch.linspace(0, 1, steps=len(raps_in_fit_range))
    y = torch.log(raps_in_fit_range)

    for _ in range(200):
        # calculate loss which will be minimised
        prediction = background_model(x).squeeze()
        difference = prediction - y
        mean_squared_error = torch.mean(difference**2)

        # backprop, step and zero gradients
        mean_squared_error.backward()
        background_optimiser.step()
        background_optimiser.zero_grad()

    # subtract background model from values
    background = torch.exp(background_model(x).squeeze())
    raps_in_fit_range -= background

    # simulate a set of 1D ctf^2 at different defoci to find best match
    defocus_step = 0.01  # microns
    test_defoci = torch.arange(
        start=defocus_range_microns[0],
        end=defocus_range_microns[1] + defocus_step,
        step=defocus_step,
    )
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

    # fit only in fitting range
    simulated_ctf2_in_fit_range = ctf2[:, fit_mask]

    # normalise experimental values in fitting range
    raps_in_fit_range_norm = torch.linalg.norm(raps_in_fit_range)
    normalised_raps_in_fit_range = raps_in_fit_range / raps_in_fit_range_norm

    # containers for outputs
    cross_correlations_2d: Optional[torch.Tensor]
    test_B_values: Optional[torch.Tensor]

    if optimize_envelope:
        # construct candidate B-factor envelopes over the same spatial frequencies
        b_low, b_high = b_range
        if b_step <= 0:
            raise ValueError("b_step must be positive.")
        test_B_values = torch.arange(
            start=b_low,
            end=b_high + b_step,
            step=b_step,
            device=ctf2.device,
            dtype=ctf2.dtype,
        )
        # envelope for power spectrum: exp(-(B * f^2) / 2)
        env_power_full = torch.exp(
            -(test_B_values[:, None] * spatial_freqs[None, :] ** 2) / 2.0
        )
        env_power_in_fit_range = env_power_full[:, fit_mask]  # (n_B, n_fit_freqs)

        # broadcast CTF^2 over B and normalise per (defocus, B)
        simulated_ctf2_in_fit_range_expanded = simulated_ctf2_in_fit_range[
            :, None, :
        ]  # (n_defocus, 1, n_fit)
        simulated_ctf2_with_env = (
            simulated_ctf2_in_fit_range_expanded * env_power_in_fit_range[None, :, :]
        )  # (n_defocus, n_B, n_fit)
        n_defocus, n_B, n_fit = simulated_ctf2_with_env.shape
        simulated_ctf2_flat = simulated_ctf2_with_env.reshape(-1, n_fit)
        simulated_ctf2_norms = torch.linalg.norm(
            simulated_ctf2_flat, dim=-1, keepdim=True
        )
        simulated_ctf2_flat = simulated_ctf2_flat / simulated_ctf2_norms

        # zero normalised cross correlation for each (defocus, B)
        zncc_flat = einops.einsum(
            simulated_ctf2_flat,
            normalised_raps_in_fit_range,
            "b i, i -> b",
        )
        zncc_2d = zncc_flat.reshape(n_defocus, n_B)
        cross_correlations_2d = zncc_2d
        cross_correlations_1d = zncc_2d.max(dim=1).values

        max_correlation_idx = torch.argmax(zncc_flat)
        best_defocus_idx = max_correlation_idx // n_B
        best_B_idx = max_correlation_idx % n_B
        best_defocus = test_defoci[best_defocus_idx]
        best_B = test_B_values[best_B_idx]
    else:
        test_B_values = None
        cross_correlations_2d = None

        # normalise simulated values in fitting range
        simulated_ctf2_norms = torch.linalg.norm(
            simulated_ctf2_in_fit_range, dim=-1, keepdim=True
        )
        simulated_ctf2_in_fit_range = simulated_ctf2_in_fit_range / simulated_ctf2_norms

        # calculate zero normalised cross correlation (defocus-only)
        zncc = einops.einsum(
            simulated_ctf2_in_fit_range,
            normalised_raps_in_fit_range,
            "b i, i -> b",
        )
        cross_correlations_1d = zncc
        max_correlation_idx = torch.argmax(zncc)
        best_defocus = test_defoci[max_correlation_idx]
        best_B = None

    return Defocus1DResults(
        frequencies_1d=fftfreq_to_spatial_frequency(freqs, pixel_spacing_angstroms),
        powerspectrum_1d=rotationally_averaged_power_spectrum,
        background_model=background_model,
        test_defoci=test_defoci,
        cross_correlations=cross_correlations_1d,
        ctf_model=CTF(
            defocus_um=best_defocus,
            voltage_kev=torch.as_tensor(voltage_kev, dtype=torch.float32),
            spherical_aberration_mm=torch.as_tensor(
                spherical_aberration_mm, dtype=torch.float32
            ),
            amplitude_contrast_fraction=torch.as_tensor(
                amplitude_contrast, dtype=torch.float32
            ),
            phase_shift_degrees=torch.as_tensor(0.0, dtype=torch.float32),
            envelope_B=best_B
            if best_B is None
            else torch.as_tensor(best_B, dtype=torch.float32),
        ),
        low_frequency_fit=1 / low_ang,
        high_frequency_fit=1 / high_ang,
        envelope_B=best_B
        if best_B is None
        else torch.as_tensor(best_B, dtype=torch.float32),
        test_B_values=test_B_values,
        cross_correlations_2d=cross_correlations_2d,
    )
