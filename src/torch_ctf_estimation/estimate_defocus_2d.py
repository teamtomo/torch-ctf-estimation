"""Estimate defocus in 2D from a power spectrum."""

import math
from typing import Any, Optional

import einops
import torch
from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.functional_serializers import SerializerFunctionWrapHandler
from torch_ctf import calculate_ctf_2d
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.envelopes import b_envelope
from torch_grid_utils.fftfreq_grid import spatial_frequency_to_fftfreq


class Defocus2DResults(BaseModel):
    """Results from 2D defocus estimation: defocus grid model and optional traces."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defocus_model: CubicCatmullRomGrid3d
    patch_power_spectra: Optional[torch.Tensor] = None
    model_trace: Optional[list[torch.Tensor]] = None
    simulated_ctf2s: Optional[torch.Tensor] = None
    astigmatism: Optional[float] = None
    astigmatism_angle: Optional[float] = None
    astigmatism_trace: Optional[list[float]] = None
    astigmatism_angle_trace: Optional[list[float]] = None
    envelope_B: Optional[float] = None
    envelope_B_trace: Optional[list[float]] = None
    loss_trace: Optional[list[float]] = None
    defocus_u: Optional[float] = None  # highest principal defocus (defocus + astig/2)
    defocus_v: Optional[float] = None  # lowest principal defocus (defocus - astig/2)

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicCatmullRomGrid3d):
            return value.to_dict()
        return handler(value)


def estimate_defocus_2d(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    defocus_grid_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus: float,
    pixel_spacing_angstroms: float,
    initial_astigmatism: float = 0.0,
    initial_astigmatism_angle: float = 0.0,
    optimize_astigmatism: bool = False,
    initial_envelope_B: float = 0.0,
    debug: bool = False,
) -> Defocus2DResults:
    """
    Estimate defocus in 2D from a power spectrum.

    Parameters
    ----------
    patch_power_spectra: torch.Tensor
        Patch power spectra.
    normalised_patch_positions: torch.Tensor
        Normalised patch positions.
    defocus_grid_resolution: tuple[int, int, int]
        Resolution of the defocus grid.
    frequency_fit_range_angstroms: tuple[float, float]
        `(low, high)` frequency fit range in angstroms.
    initial_defocus: float
        Initial defocus in microns.
    pixel_spacing_angstroms: float
        Isotropic pixel spacing in angstroms.
    initial_astigmatism: float
        Initial astigmatism in microns.
    initial_astigmatism_angle: float
        Initial astigmatism angle in degrees.
    optimize_astigmatism: bool
        Whether to optimize the astigmatism.
    initial_envelope_B: float
        Initial B-factor for envelope.
    debug: bool
        Whether to return debug information.

    Returns
    -------
    Defocus2DResults
        Results from 2D defocus estimation containing defocus model,
        astigmatism, astigmatism angle, envelope B, and loss trace.
    """
    # grab patch sidelength
    patch_sidelength = patch_power_spectra.shape[-2]

    # if only 1 grid point in t, take mean of all patches
    nt, nh, nw = defocus_grid_resolution
    if nt == 1:
        patch_power_spectra = einops.reduce(
            patch_power_spectra, "t ... -> 1 ...", reduction="mean"
        )

    # Initialise defocus model as 3D grid with defined resolution at initial defocus
    defocus_grid_data = torch.ones(size=defocus_grid_resolution) * initial_defocus
    defocus_model = CubicCatmullRomGrid3d.from_grid_data(defocus_grid_data)

    device = patch_power_spectra.device
    # Initialize astigmatism parameters
    # Angle is parameterized as (angle_u, angle_v) on unit circle to avoid wrap-around
    _angle_rad = initial_astigmatism_angle * math.pi / 180.0
    _angle_u_init = math.cos(_angle_rad)
    _angle_v_init = math.sin(_angle_rad)
    if optimize_astigmatism:
        # Start from a non-trivial value when 0 so optimizer can move it
        init_astig = initial_astigmatism if initial_astigmatism > 0 else 0.05
        astigmatism = torch.nn.Parameter(torch.tensor(init_astig, device=device))
        angle_u = torch.nn.Parameter(torch.tensor(_angle_u_init, device=device))
        angle_v = torch.nn.Parameter(torch.tensor(_angle_v_init, device=device))
    else:
        astigmatism = torch.tensor(initial_astigmatism, device=device)
        angle_u = torch.tensor(_angle_u_init, device=device)
        angle_v = torch.tensor(_angle_v_init, device=device)

    # Fixed B-factor from 1D estimation (used for envelope in simulated CTF^2)
    envelope_B = torch.tensor(initial_envelope_B, device=device)

    # bandpass data to fit range
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(
        1 / low_ang, spacing=pixel_spacing_angstroms
    )
    high_fftfreq = spatial_frequency_to_fftfreq(
        1 / high_ang, spacing=pixel_spacing_angstroms
    )
    bp_filter = bandpass_filter(
        low=low_fftfreq,
        high=high_fftfreq,
        falloff=0,
        image_shape=(patch_sidelength, patch_sidelength),
        rfft=True,
        fftshift=False,
        device=patch_power_spectra.device,
    )
    patch_power_spectra *= bp_filter

    # optimise 2d+t defocus model, optionally astigmatism and astigmatism_angle
    param_groups = [{"params": defocus_model.parameters(), "lr": 0.01}]
    if optimize_astigmatism:
        param_groups.extend(
            [
                {"params": [astigmatism], "lr": 0.05},
                {"params": [angle_u, angle_v], "lr": 50.0},
            ]
        )

    optimiser = torch.optim.Adam(params=param_groups)

    defocus_models = []
    astigmatism_trace: list[float] = []
    astigmatism_angle_trace: list[float] = []
    loss_trace: list[float] = []
    env_2d = b_envelope(
        B=envelope_B,
        image_shape=(patch_sidelength, patch_sidelength),
        pixel_size=pixel_spacing_angstroms,
        rfft=True,
        fftshift=False,
        device=patch_power_spectra.device,
    )
    T = patch_power_spectra.shape[0]
    simulated_ctf2s = None  # for debug: last t's value from final iteration

    for _ in range(100):
        # Check astigmatism parameters for NaN before using them
        if optimize_astigmatism:
            if (
                torch.isnan(astigmatism)
                or torch.isnan(angle_u)
                or torch.isnan(angle_v)
                or torch.isinf(astigmatism)
                or torch.isinf(angle_u)
                or torch.isinf(angle_v)
            ):
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                continue

            # Clamp for CTF: astigmatism >= 1e-6 (avoid zero so gradient flows)
            astig_clamped = torch.clamp(astigmatism, min=1e-6)
            # Angle from unit-circle (u,v): atan2(v,u) in degrees, then [0, 180]
            _eps = 1e-8
            _norm = torch.sqrt(angle_u**2 + angle_v**2 + _eps)
            _dir_u = angle_u / _norm
            _dir_v = angle_v / _norm
            _angle_rad = torch.atan2(_dir_v, _dir_u)
            _angle_deg = _angle_rad * (180.0 / math.pi)
            astig_angle_clamped = torch.remainder(_angle_deg + 180.0, 180.0)
        else:
            astig_clamped = astigmatism
            _angle_rad = torch.atan2(angle_v, angle_u)
            _angle_deg = _angle_rad * (180.0 / math.pi)
            astig_angle_clamped = torch.remainder(_angle_deg + 180.0, 180.0)

        optimiser.zero_grad()
        loss_t_list: list[torch.Tensor] = []

        for t_idx in range(T):
            patch_ps_t = patch_power_spectra[t_idx]
            positions_t = normalised_patch_positions[t_idx]

            predicted_defocus_t = defocus_model(positions_t)
            predicted_defocus_t = einops.rearrange(predicted_defocus_t, "... 1 -> ...")

            simulated_ctf2s_t = (
                calculate_ctf_2d(
                    defocus=predicted_defocus_t,
                    voltage=300,
                    spherical_aberration=2.7,
                    amplitude_contrast=0.10,
                    phase_shift=0,
                    pixel_size=pixel_spacing_angstroms,
                    image_shape=(patch_sidelength, patch_sidelength),
                    astigmatism=astig_clamped,
                    astigmatism_angle=astig_angle_clamped,
                    rfft=True,
                    fftshift=False,
                )
                ** 2
            )
            simulated_ctf2s_t = simulated_ctf2s_t * (env_2d**2)
            simulated_ctf2s_t = simulated_ctf2s_t * bp_filter
            simulated_ctf2s = simulated_ctf2s_t  # debug: keep last t

            if (
                torch.isnan(simulated_ctf2s_t).any()
                or torch.isinf(simulated_ctf2s_t).any()
            ):
                continue  # skip this t's backward

            model_flat = simulated_ctf2s_t.reshape(-1)
            data_flat = patch_ps_t.reshape(-1)
            eps = 1e-8
            model_norm = (model_flat - model_flat.mean()) / (model_flat.std() + eps)
            data_norm = (data_flat - data_flat.mean()) / (data_flat.std() + eps)
            C_t = (model_norm * data_norm).sum()
            loss_t = -C_t
            (loss_t / T).backward()
            loss_t_list.append(loss_t.detach())

        if len(loss_t_list) == 0:
            if optimize_astigmatism:
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
            continue

        mean_loss = sum(loss_t_list) / len(loss_t_list)
        if torch.isnan(mean_loss) or torch.isinf(mean_loss):
            if optimize_astigmatism:
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
            continue

        loss_trace.append(float(mean_loss.cpu().item()))

        if optimize_astigmatism:
            if astigmatism.grad is not None and (
                torch.isnan(astigmatism.grad) or torch.isinf(astigmatism.grad)
            ):
                optimiser.zero_grad()
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                continue
            if (
                angle_u.grad is not None
                and (torch.isnan(angle_u.grad) or torch.isinf(angle_u.grad))
            ) or (
                angle_v.grad is not None
                and (torch.isnan(angle_v.grad) or torch.isinf(angle_v.grad))
            ):
                optimiser.zero_grad()
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                continue

        optimiser.step()

        # Constrain astigmatism after optimization step and check for NaN
        if optimize_astigmatism:
            with torch.no_grad():
                # Check for NaN after step and reset if found
                if (
                    torch.isnan(astigmatism)
                    or torch.isnan(angle_u)
                    or torch.isnan(angle_v)
                    or torch.isinf(astigmatism)
                    or torch.isinf(angle_u)
                    or torch.isinf(angle_v)
                ):
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                else:
                    astigmatism.clamp_(min=1e-6)

        defocus_models.append(defocus_model.data.detach().clone())
        if optimize_astigmatism:
            astigmatism_trace.append(float(astigmatism.detach().cpu().item()))
            # Record angle in degrees [0, 180] from current (angle_u, angle_v)
            _norm = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
            _a_rad = torch.atan2(angle_v.detach() / _norm, angle_u.detach() / _norm)
            _a_deg = float((_a_rad * (180.0 / math.pi) + 180.0) % 180.0)
            astigmatism_angle_trace.append(_a_deg)

    final_astigmatism = float(astigmatism.detach().cpu().item())
    _fn = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
    _fa_rad = torch.atan2(angle_v.detach() / _fn, angle_u.detach() / _fn)
    final_astigmatism_angle = float(
        (_fa_rad.cpu().item() * (180.0 / math.pi) + 180.0) % 180.0
    )
    final_envelope_B = float(envelope_B.detach().cpu().item())

    # Principal defoci from mean defocus and astigmatism
    mean_defocus = float(defocus_model.data.detach().cpu().mean().item())
    final_defocus_u = mean_defocus + final_astigmatism / 2.0
    final_defocus_v = mean_defocus - final_astigmatism / 2.0

    if debug:
        return Defocus2DResults(
            defocus_model=defocus_model,
            simulated_ctf2s=simulated_ctf2s,
            patch_power_spectra=patch_power_spectra,
            model_trace=defocus_models,
            astigmatism=final_astigmatism,
            astigmatism_angle=final_astigmatism_angle,
            astigmatism_trace=astigmatism_trace if optimize_astigmatism else None,
            astigmatism_angle_trace=astigmatism_angle_trace
            if optimize_astigmatism
            else None,
            envelope_B=final_envelope_B,
            envelope_B_trace=None,
            loss_trace=loss_trace,
            defocus_u=final_defocus_u,
            defocus_v=final_defocus_v,
        )
    else:
        return Defocus2DResults(
            defocus_model=defocus_model,
            astigmatism=final_astigmatism,
            astigmatism_angle=final_astigmatism_angle,
            envelope_B=final_envelope_B,
            defocus_u=final_defocus_u,
            defocus_v=final_defocus_v,
        )
