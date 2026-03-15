"""CTF simulation and correlation loss for 2D defocus estimation."""

from typing import Optional

import torch
from torch_ctf import calc_LPP_ctf_2D, calculate_ctf_2d

from torch_ctf_estimation.models import LaserParams

# Penalty weight for unit-circle constraint on (u,v): lambda*(u^2+v^2-1)^2
PHASE_SHIFT_UNIT_CIRCLE_PENALTY = 0.1


def compute_ctf2_t(
    defocus_t: torch.Tensor,
    phase_shift_t: torch.Tensor,
    astig_clamped: torch.Tensor,
    astig_angle_clamped: torch.Tensor,
    image_shape: tuple[int, int],
    pixel_spacing_angstroms: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast_fraction: float,
    env_2d: torch.Tensor,
    bp_filter: torch.Tensor,
    laser_params: Optional[LaserParams] = None,
) -> torch.Tensor:
    """
    Compute CTF^2 * env^2 * bp_filter for one frame (patch grid).

    Returns simulated power spectrum in same shape as defocus_t (rfft layout).
    """
    if laser_params is not None:
        ctf_t = calc_LPP_ctf_2D(
            defocus=defocus_t,
            astigmatism=astig_clamped,
            astigmatism_angle=astig_angle_clamped,
            voltage=voltage_kev,
            spherical_aberration=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast_fraction,
            pixel_size=pixel_spacing_angstroms,
            image_shape=image_shape,
            rfft=True,
            fftshift=False,
            NA=laser_params.NA,
            laser_wavelength_angstrom=laser_params.laser_wavelength_angstrom,
            focal_length_angstrom=laser_params.focal_length_angstrom,
            laser_xy_angle_deg=laser_params.laser_xy_angle_deg,
            laser_xz_angle_deg=laser_params.laser_xz_angle_deg,
            laser_long_offset_angstrom=laser_params.laser_long_offset_angstrom,
            laser_trans_offset_angstrom=laser_params.laser_trans_offset_angstrom,
            laser_polarization_angle_deg=laser_params.laser_polarization_angle_deg,
            peak_phase_deg=laser_params.peak_phase_deg,
            dual_laser=laser_params.dual_laser,
            beam_tilt_mrad=None,
            even_zernike_coeffs=None,
            odd_zernike_coeffs=None,
            transform_matrix=None,
        )
    else:
        ctf_t = calculate_ctf_2d(
            defocus=defocus_t,
            voltage=voltage_kev,
            spherical_aberration=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast_fraction,
            phase_shift=phase_shift_t,
            pixel_size=pixel_spacing_angstroms,
            image_shape=image_shape,
            astigmatism=astig_clamped,
            astigmatism_angle=astig_angle_clamped,
            rfft=True,
            fftshift=False,
        )
    simulated_ctf2s_t = ctf_t**2
    simulated_ctf2s_t = simulated_ctf2s_t * (env_2d**2) * bp_filter
    return simulated_ctf2s_t


def correlation_loss_t(
    simulated_ctf2s_t: torch.Tensor,
    patch_ps_t: torch.Tensor,
    u_t: Optional[torch.Tensor] = None,
    v_t: Optional[torch.Tensor] = None,
    phase_penalty_weight: float = PHASE_SHIFT_UNIT_CIRCLE_PENALTY,
) -> torch.Tensor:
    """
    Normalised correlation loss (-ZNCC) plus optional (u,v) unit-circle penalty.

    Caller should ensure simulated_ctf2s_t has no NaN/Inf before calling.
    """
    model_flat = simulated_ctf2s_t.reshape(-1)
    data_flat = patch_ps_t.reshape(-1)
    eps = 1e-8
    model_norm = (model_flat - model_flat.mean()) / (model_flat.std() + eps)
    data_norm = (data_flat - data_flat.mean()) / (data_flat.std() + eps)
    C_t = (model_norm * data_norm).sum()
    loss_t = -C_t
    if u_t is not None and v_t is not None:
        penalty_t = ((u_t**2 + v_t**2 - 1.0) ** 2).mean()
        loss_t = loss_t + phase_penalty_weight * penalty_t
    return loss_t
