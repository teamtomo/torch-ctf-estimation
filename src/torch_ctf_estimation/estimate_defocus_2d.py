"""Estimate defocus in 2D from a power spectrum."""

import math
from typing import Any, Literal, Optional, Union

import einops
import torch
from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.functional_serializers import SerializerFunctionWrapHandler
from torch_ctf import calculate_ctf_2d
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.envelopes import b_envelope
from torch_grid_utils.fftfreq_grid import spatial_frequency_to_fftfreq

from torch_ctf_estimation.models import LinearDefocusModel, QuadraticPhaseShiftModel

# Penalty weight for unit-circle constraint on (u,v): lambda*(u^2+v^2-1)^2
PHASE_SHIFT_UNIT_CIRCLE_PENALTY = 0.1


def _astig_angle_to_m90_p90(angle_0_180: float) -> float:
    """Map astigmatism angle from [0, 180) to [-90, 90] for output."""
    a = angle_0_180 % 180.0
    return a if a <= 90.0 else a - 180.0


try:
    from torch_cubic_spline_grids import CubicCatmullRomGrid1d
except ImportError:
    CubicCatmullRomGrid1d = None


class Defocus2DResults(BaseModel):
    """2D defocus result: defocus model (grid or linear) and optional traces."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defocus_model_type: Literal["grid", "linear"] = "grid"
    defocus_model: Union[CubicCatmullRomGrid3d, LinearDefocusModel]
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
    tilt_axis_angle_deg: Optional[float] = (
        None  # set when linear + pixel_spacing/size known
    )
    tilt_magnitude_deg: Optional[float] = None
    # Phase shift (when optimize_phase_shift=True)
    phase_shift_degrees: Optional[float] = None
    phase_shift_model_type: Optional[Literal["grid", "quadratic"]] = None
    phase_shift_model: Optional[
        Union[
            CubicCatmullRomGrid3d,
            QuadraticPhaseShiftModel,
            tuple[CubicCatmullRomGrid3d, CubicCatmullRomGrid3d],
        ]
    ] = None
    phase_shift_trace: Optional[list[float]] = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicCatmullRomGrid3d):
            return value.to_dict()
        if isinstance(value, LinearDefocusModel):
            return value.model_dump()
        if isinstance(value, QuadraticPhaseShiftModel):
            return value.model_dump()
        if isinstance(value, tuple) and len(value) == 2:
            a, b = value
            if isinstance(a, CubicCatmullRomGrid3d) and isinstance(
                b, CubicCatmullRomGrid3d
            ):
                return {"u": a.to_dict(), "v": b.to_dict()}
        return handler(value)

    def linear_tilt_axis_and_magnitude_deg(
        self,
        pixel_spacing_angstroms: float,
        image_size_pixels: int,
    ) -> tuple[float, float]:
        """
        For linear defocus model: (tilt_axis_angle_deg, tilt_magnitude_deg).

        Tilt axis angle is the defocus gradient direction (degrees).
        Tilt magnitude (degrees) from defocus gradient (um per normalized unit)
        and pixel size: arctan(gradient in µm/µm) with gradient in µm/µm
        = grad_mag*1e4 / (image_size_pixels * pixel_spacing_angstroms).
        Returns (0.0, 0.0) if defocus_model_type is not "linear".
        """
        return linear_tilt_axis_and_magnitude_deg(
            self, pixel_spacing_angstroms, image_size_pixels
        )


def linear_tilt_axis_and_magnitude_deg(
    result2d: Defocus2DResults,
    pixel_spacing_angstroms: float,
    image_size_pixels: int,
) -> tuple[float, float]:
    """
    Tilt axis angle (degrees) and tilt magnitude (degrees) from a linear defocus result.

    The defocus gradient is in um per normalized unit (0-1 across the image).
    Tilt magnitude (deg) = arctan(gradient in um/um) * 180/pi, with
    gradient in µm/µm = defocus_gradient_magnitude * 1e4
    / (image_size_pixels * pixel_spacing_angstroms)
    (1 normalized unit = image_size_pixels * pixel_spacing_angstroms A).

    Returns (tilt_axis_angle_deg, tilt_magnitude_deg). If defocus_model_type is not
    "linear", returns (0.0, 0.0).
    """
    if result2d.defocus_model_type != "linear":
        return (0.0, 0.0)
    lm = result2d.defocus_model
    grad_mag = lm.defocus_gradient_magnitude  # microns per normalized unit
    axis_deg = lm.defocus_gradient_angle
    # 1 norm unit = image_size_pixels * pixel_spacing_angstroms Angstroms
    # 1 Angstrom = 1e-4 um => gradient in um/um = grad_mag*1e4/(N*pixel_size_Ang)
    grad_per_micron = (grad_mag * 1e4) / (image_size_pixels * pixel_spacing_angstroms)
    tilt_rad = math.atan(grad_per_micron)
    tilt_deg = tilt_rad * 180.0 / math.pi
    return (axis_deg, tilt_deg)


def _shared_astigmatism_and_env(
    *,
    patch_power_spectra: torch.Tensor,
    image_shape: tuple[int, int],
    device: torch.device,
    frequency_fit_range_angstroms: tuple[float, float],
    pixel_spacing_angstroms: float,
    initial_astigmatism: float,
    initial_astigmatism_angle: float,
    optimize_astigmatism: bool,
    initial_envelope_B: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    torch.Tensor,
    torch.Tensor,
]:
    """Build bandpass filter, astigmatism params, and B-factor envelope."""
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
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        device=device,
    )
    _angle_rad = initial_astigmatism_angle * math.pi / 180.0
    _angle_u_init = math.cos(_angle_rad)
    _angle_v_init = math.sin(_angle_rad)
    if optimize_astigmatism:
        init_astig = initial_astigmatism if initial_astigmatism > 0 else 0.05
        astigmatism = torch.nn.Parameter(torch.tensor(init_astig, device=device))
        angle_u = torch.nn.Parameter(torch.tensor(_angle_u_init, device=device))
        angle_v = torch.nn.Parameter(torch.tensor(_angle_v_init, device=device))
    else:
        astigmatism = torch.tensor(initial_astigmatism, device=device)
        angle_u = torch.tensor(_angle_u_init, device=device)
        angle_v = torch.tensor(_angle_v_init, device=device)
    envelope_B = torch.tensor(initial_envelope_B, device=device)
    env_2d = b_envelope(
        B=envelope_B,
        image_shape=image_shape,
        pixel_size=pixel_spacing_angstroms,
        rfft=True,
        fftshift=False,
        device=device,
    )
    return (
        bp_filter,
        astigmatism,
        angle_u,
        angle_v,
        _angle_u_init,
        _angle_v_init,
        envelope_B,
        env_2d,
    )


def estimate_defocus_2d_grid(
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
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    astigmatism_lr: float = 0.05,
    astigmatism_angle_lr: float = 50.0,
    debug: bool = False,
    optimize_phase_shift: bool = False,
    phase_shift_model: Literal["grid", "quadratic"] = "grid",
    initial_phase_shift: float = 0.0,
    phase_shift_lr: float = 5.0,
) -> Defocus2DResults:
    """
    Estimate defocus in 2D using a 3D spline grid over (t, x, y).

    See :func:`estimate_defocus_2d` for parameter descriptions.
    """
    # Derive spatial image shape from rfft PS (H, W_rfft) -> (H, (W_rfft-1)*2)
    ph, pw_rfft = patch_power_spectra.shape[-2], patch_power_spectra.shape[-1]
    image_shape = (ph, (pw_rfft - 1) * 2)
    device = patch_power_spectra.device
    nt, _, _ = defocus_grid_resolution
    if nt == 1:
        patch_power_spectra = einops.reduce(
            patch_power_spectra, "t ... -> 1 ...", reduction="mean"
        )

    defocus_grid_data = (
        torch.ones(size=defocus_grid_resolution, device=device) * initial_defocus
    )
    defocus_model_obj = CubicCatmullRomGrid3d.from_grid_data(defocus_grid_data).to(
        device
    )
    phase_shift_u_grid_model = None
    phase_shift_v_grid_model = None
    phase_shift_quad_params = None
    if optimize_phase_shift:
        if phase_shift_model == "grid":
            theta_rad = initial_phase_shift * (math.pi / 180.0)
            u_init = math.cos(2.0 * theta_rad)
            v_init = math.sin(2.0 * theta_rad)
            phase_shift_u_grid_data = (
                torch.ones(size=defocus_grid_resolution, device=device) * u_init
            )
            phase_shift_v_grid_data = (
                torch.ones(size=defocus_grid_resolution, device=device) * v_init
            )
            phase_shift_u_grid_model = CubicCatmullRomGrid3d.from_grid_data(
                phase_shift_u_grid_data
            ).to(device)
            phase_shift_v_grid_model = CubicCatmullRomGrid3d.from_grid_data(
                phase_shift_v_grid_data
            ).to(device)
        else:
            # quadratic: f(x,y)=C+g*s+k*s^2, s=x*cos(alpha)+y*sin(alpha).
            # 4 params: C, g, k, alpha.
            phase_shift_quad_params = {
                "C": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift, device=device, dtype=torch.float32
                    )
                ),
                "g": torch.nn.Parameter(
                    torch.tensor(0.0, device=device, dtype=torch.float32)
                ),
                "k": torch.nn.Parameter(
                    torch.tensor(0.0, device=device, dtype=torch.float32)
                ),
                "alpha": torch.nn.Parameter(
                    torch.tensor(0.0, device=device, dtype=torch.float32)
                ),
            }

    (
        bp_filter,
        astigmatism,
        angle_u,
        angle_v,
        _angle_u_init,
        _angle_v_init,
        envelope_B,
        env_2d,
    ) = _shared_astigmatism_and_env(
        patch_power_spectra=patch_power_spectra,
        image_shape=image_shape,
        device=device,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=initial_astigmatism,
        initial_astigmatism_angle=initial_astigmatism_angle,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
    )
    patch_power_spectra = patch_power_spectra * bp_filter

    param_groups = [{"params": defocus_model_obj.parameters(), "lr": defocus_lr}]
    if optimize_astigmatism:
        param_groups.extend(
            [
                {"params": [astigmatism], "lr": astigmatism_lr},
                {"params": [angle_u, angle_v], "lr": astigmatism_angle_lr},
            ]
        )
    if phase_shift_u_grid_model is not None and phase_shift_v_grid_model is not None:
        param_groups.append(
            {"params": phase_shift_u_grid_model.parameters(), "lr": phase_shift_lr}
        )
        param_groups.append(
            {"params": phase_shift_v_grid_model.parameters(), "lr": phase_shift_lr}
        )
    if phase_shift_quad_params is not None:
        param_groups.append(
            {
                "params": list(phase_shift_quad_params.values()),
                "lr": phase_shift_lr,
            }
        )
    optimiser = torch.optim.Adam(params=param_groups)

    defocus_models: list[torch.Tensor] = []
    astigmatism_trace: list[float] = []
    astigmatism_angle_trace: list[float] = []
    phase_shift_trace: list[float] = []
    loss_trace: list[float] = []
    T = patch_power_spectra.shape[0]
    simulated_ctf2s = None

    for _ in range(n_iterations):
        if optimize_astigmatism:
            if (
                torch.isnan(astigmatism).any()
                or torch.isnan(angle_u).any()
                or torch.isnan(angle_v).any()
                or torch.isinf(astigmatism).any()
                or torch.isinf(angle_u).any()
                or torch.isinf(angle_v).any()
            ):
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                continue
            astig_clamped = torch.clamp(astigmatism, min=1e-6)
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
            predicted_defocus_t = defocus_model_obj(positions_t)
            predicted_defocus_t = einops.rearrange(predicted_defocus_t, "... 1 -> ...")
            u_t = v_t = None
            if (
                phase_shift_u_grid_model is not None
                and phase_shift_v_grid_model is not None
            ):
                u_t = phase_shift_u_grid_model(positions_t).squeeze(-1)
                v_t = phase_shift_v_grid_model(positions_t).squeeze(-1)
                phase_shift_t = torch.remainder(
                    0.5 * torch.atan2(v_t, u_t) * (180.0 / math.pi), 180.0
                )
            elif phase_shift_quad_params is not None:
                # f(x,y) = C + g*s + k*s^2, s = x*cos(alpha)+y*sin(alpha)
                x = 2.0 * positions_t[..., 1] - 1.0
                y = 2.0 * positions_t[..., 2] - 1.0
                alpha = phase_shift_quad_params["alpha"]
                s = x * torch.cos(alpha) + y * torch.sin(alpha)
                phase_shift_t = (
                    phase_shift_quad_params["C"]
                    + phase_shift_quad_params["g"] * s
                    + phase_shift_quad_params["k"] * (s**2)
                )
                phase_shift_t = torch.clamp(phase_shift_t, min=0.0, max=180.0)
            else:
                phase_shift_t = 0
            simulated_ctf2s_t = (
                calculate_ctf_2d(
                    defocus=predicted_defocus_t,
                    voltage=300,
                    spherical_aberration=2.7,
                    amplitude_contrast=0.10,
                    phase_shift=phase_shift_t,
                    pixel_size=pixel_spacing_angstroms,
                    image_shape=image_shape,
                    astigmatism=astig_clamped,
                    astigmatism_angle=astig_angle_clamped,
                    rfft=True,
                    fftshift=False,
                )
                ** 2
            )
            simulated_ctf2s_t = simulated_ctf2s_t * (env_2d**2) * bp_filter
            simulated_ctf2s = simulated_ctf2s_t
            if (
                torch.isnan(simulated_ctf2s_t).any()
                or torch.isinf(simulated_ctf2s_t).any()
            ):
                continue
            model_flat = simulated_ctf2s_t.reshape(-1)
            data_flat = patch_ps_t.reshape(-1)
            eps = 1e-8
            model_norm = (model_flat - model_flat.mean()) / (model_flat.std() + eps)
            data_norm = (data_flat - data_flat.mean()) / (data_flat.std() + eps)
            C_t = (model_norm * data_norm).sum()
            loss_t = -C_t
            if u_t is not None and v_t is not None:
                penalty_t = ((u_t**2 + v_t**2 - 1.0) ** 2).mean()
                loss_t = loss_t + PHASE_SHIFT_UNIT_CIRCLE_PENALTY * penalty_t
            (loss_t / T).backward()
            loss_t_list.append(loss_t.detach())

        if (
            phase_shift_u_grid_model is not None
            and phase_shift_v_grid_model is not None
        ):
            pass  # no clamp: (u,v) unit circle handled by penalty
        if phase_shift_quad_params is not None:
            with torch.no_grad():
                phase_shift_quad_params["C"].clamp_(min=0.0, max=180.0)

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
        if optimize_astigmatism:
            with torch.no_grad():
                if (
                    torch.isnan(astigmatism).any()
                    or torch.isnan(angle_u).any()
                    or torch.isnan(angle_v).any()
                    or torch.isinf(astigmatism).any()
                    or torch.isinf(angle_u).any()
                    or torch.isinf(angle_v).any()
                ):
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                else:
                    astigmatism.clamp_(min=1e-6)
        defocus_models.append(defocus_model_obj.data.detach().clone())
        if optimize_astigmatism:
            astigmatism_trace.append(float(astigmatism.detach().cpu().item()))
            _norm = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
            _a_rad = torch.atan2(angle_v.detach() / _norm, angle_u.detach() / _norm)
            _a_deg = float((_a_rad * (180.0 / math.pi) + 180.0) % 180.0)
            astigmatism_angle_trace.append(_astig_angle_to_m90_p90(_a_deg))
        if (
            phase_shift_u_grid_model is not None
            and phase_shift_v_grid_model is not None
        ):
            _mu = phase_shift_u_grid_model.data.detach().cpu().mean().item()
            _mv = phase_shift_v_grid_model.data.detach().cpu().mean().item()
            _p = (0.5 * math.degrees(math.atan2(_mv, _mu))) % 180.0
            phase_shift_trace.append(min(_p, 180.0 - _p))
        if phase_shift_quad_params is not None:
            _c = float(phase_shift_quad_params["C"].detach().cpu().item())
            phase_shift_trace.append(min(_c, 180.0 - _c))

    final_astigmatism = float(astigmatism.detach().cpu().item())
    _fn = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
    _fa_rad = torch.atan2(angle_v.detach() / _fn, angle_u.detach() / _fn)
    _fa_deg = float((_fa_rad.cpu().item() * (180.0 / math.pi) + 180.0) % 180.0)
    final_astigmatism_angle = _astig_angle_to_m90_p90(_fa_deg)
    final_envelope_B = float(envelope_B.detach().cpu().item())
    mean_defocus = float(defocus_model_obj.data.detach().cpu().mean().item())
    final_defocus_u = mean_defocus + final_astigmatism / 2.0
    final_defocus_v = mean_defocus - final_astigmatism / 2.0
    final_phase_shift_deg = None
    final_phase_shift_model_obj = None
    if phase_shift_u_grid_model is not None and phase_shift_v_grid_model is not None:
        _mu = phase_shift_u_grid_model.data.detach().cpu().mean().item()
        _mv = phase_shift_v_grid_model.data.detach().cpu().mean().item()
        _p = (0.5 * math.degrees(math.atan2(_mv, _mu))) % 180.0
        final_phase_shift_deg = min(_p, 180.0 - _p)
        final_phase_shift_model_obj = (
            phase_shift_u_grid_model,
            phase_shift_v_grid_model,
        )
    elif phase_shift_quad_params is not None:
        _c = float(phase_shift_quad_params["C"].detach().cpu().item())
        final_phase_shift_deg = min(_c, 180.0 - _c)
        final_phase_shift_model_obj = QuadraticPhaseShiftModel(
            C=float(phase_shift_quad_params["C"].detach().cpu().item()),
            g=float(phase_shift_quad_params["g"].detach().cpu().item()),
            k=float(phase_shift_quad_params["k"].detach().cpu().item()),
            alpha_rad=float(phase_shift_quad_params["alpha"].detach().cpu().item()),
        )

    if debug:
        return Defocus2DResults(
            defocus_model_type="grid",
            defocus_model=defocus_model_obj,
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
            loss_trace=loss_trace,
            defocus_u=final_defocus_u,
            defocus_v=final_defocus_v,
            phase_shift_degrees=final_phase_shift_deg,
            phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
            phase_shift_model=final_phase_shift_model_obj
            if optimize_phase_shift
            else None,
            phase_shift_trace=phase_shift_trace if optimize_phase_shift else None,
        )
    return Defocus2DResults(
        defocus_model_type="grid",
        defocus_model=defocus_model_obj,
        astigmatism=final_astigmatism,
        astigmatism_angle=final_astigmatism_angle,
        envelope_B=final_envelope_B,
        defocus_u=final_defocus_u,
        defocus_v=final_defocus_v,
        phase_shift_degrees=final_phase_shift_deg,
        phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
        phase_shift_model=final_phase_shift_model_obj if optimize_phase_shift else None,
        phase_shift_trace=phase_shift_trace if optimize_phase_shift else None,
    )


def estimate_defocus_2d_linear(
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
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    astigmatism_lr: float = 0.05,
    astigmatism_angle_lr: float = 50.0,
    initial_defocus_gradient_magnitude: float = 0.0,
    initial_defocus_gradient_angle: float = 0.0,
    defocus_gradient_magnitude_lr: float = 0.05,
    defocus_gradient_angle_lr: float = 50.0,
    fix_defocus_0: Optional[float] = None,
    debug: bool = False,
    optimize_phase_shift: bool = False,
    phase_shift_model: Literal["grid", "quadratic"] = "grid",
    initial_phase_shift: float = 0.0,
    phase_shift_lr: float = 5.0,
) -> Defocus2DResults:
    """
    Estimate defocus in 2D using a linear (tilt) model in (x, y).

    Optional cubic spline in t when nt > 1. Only the first element of
    defocus_grid_resolution (nt) is used.
    See :func:`estimate_defocus_2d` for other parameter descriptions.
    """
    # Derive spatial image shape from rfft PS (H, W_rfft) -> (H, (W_rfft-1)*2)
    ph, pw_rfft = patch_power_spectra.shape[-2], patch_power_spectra.shape[-1]
    image_shape = (ph, (pw_rfft - 1) * 2)
    device = patch_power_spectra.device
    nt, _, _ = defocus_grid_resolution
    if nt == 1:
        patch_power_spectra = einops.reduce(
            patch_power_spectra, "t ... -> 1 ...", reduction="mean"
        )

    phase_shift_u_grid_model = None
    phase_shift_v_grid_model = None
    phase_shift_quad_params = None
    if optimize_phase_shift:
        if phase_shift_model == "grid":
            theta_rad = initial_phase_shift * (math.pi / 180.0)
            u_init = math.cos(2.0 * theta_rad)
            v_init = math.sin(2.0 * theta_rad)
            phase_shift_u_grid_data = (
                torch.ones(size=defocus_grid_resolution, device=device) * u_init
            )
            phase_shift_v_grid_data = (
                torch.ones(size=defocus_grid_resolution, device=device) * v_init
            )
            phase_shift_u_grid_model = CubicCatmullRomGrid3d.from_grid_data(
                phase_shift_u_grid_data
            ).to(device)
            phase_shift_v_grid_model = CubicCatmullRomGrid3d.from_grid_data(
                phase_shift_v_grid_data
            ).to(device)
        else:
            # quadratic: f(x,y)=C+g*s+k*s^2, s=x*cos(alpha)+y*sin(alpha).
            # 4 params: C, g, k, alpha.
            phase_shift_quad_params = {
                "C": torch.nn.Parameter(
                    torch.tensor(
                        initial_phase_shift, device=device, dtype=torch.float32
                    )
                ),
                "g": torch.nn.Parameter(
                    torch.tensor(0.0, device=device, dtype=torch.float32)
                ),
                "k": torch.nn.Parameter(
                    torch.tensor(0.0, device=device, dtype=torch.float32)
                ),
                "alpha": torch.nn.Parameter(
                    torch.tensor(0.0, device=device, dtype=torch.float32)
                ),
            }

    _grad_angle_rad = initial_defocus_gradient_angle * math.pi / 180.0
    _grad_angle_u_init = math.cos(_grad_angle_rad)
    _grad_angle_v_init = math.sin(_grad_angle_rad)
    init_grad_mag = (
        initial_defocus_gradient_magnitude
        if initial_defocus_gradient_magnitude != 0
        else 0.05
    )
    defocus_0_fixed = fix_defocus_0
    defocus_0_param = None
    grad_mag_param = None
    grad_angle_u = None
    grad_angle_v = None
    defocus_0_spline_1d = None
    grad_mag_spline_1d = None
    grad_angle_u_spline_1d = None
    grad_angle_v_spline_1d = None
    use_linear_splines = nt > 1 and CubicCatmullRomGrid1d is not None
    if use_linear_splines:
        if defocus_0_fixed is not None:
            defocus_0_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
                torch.ones(nt, device=device) * defocus_0_fixed
            ).to(device)
            for p in defocus_0_spline_1d.parameters():
                p.requires_grad = False
        else:
            defocus_0_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
                torch.ones(nt, device=device) * initial_defocus
            ).to(device)
        grad_mag_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
            torch.ones(nt, device=device) * init_grad_mag
        ).to(device)
        grad_angle_u_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
            torch.ones(nt, device=device) * _grad_angle_u_init
        ).to(device)
        grad_angle_v_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
            torch.ones(nt, device=device) * _grad_angle_v_init
        ).to(device)
    else:
        if defocus_0_fixed is None:
            defocus_0_param = torch.nn.Parameter(
                torch.tensor(initial_defocus, device=device, dtype=torch.float32)
            ).to(device)
        grad_mag_param = torch.nn.Parameter(
            torch.tensor(init_grad_mag, device=device, dtype=torch.float32)
        ).to(device)
        grad_angle_u = torch.nn.Parameter(
            torch.tensor(_grad_angle_u_init, device=device, dtype=torch.float32)
        ).to(device)
        grad_angle_v = torch.nn.Parameter(
            torch.tensor(_grad_angle_v_init, device=device, dtype=torch.float32)
        ).to(device)

    (
        bp_filter,
        astigmatism,
        angle_u,
        angle_v,
        _angle_u_init,
        _angle_v_init,
        envelope_B,
        env_2d,
    ) = _shared_astigmatism_and_env(
        patch_power_spectra=patch_power_spectra,
        image_shape=image_shape,
        device=device,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=initial_astigmatism,
        initial_astigmatism_angle=initial_astigmatism_angle,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
    )
    patch_power_spectra = patch_power_spectra * bp_filter

    param_groups = []
    if defocus_0_param is not None:
        param_groups.extend(
            [
                {"params": [defocus_0_param], "lr": defocus_lr},
                {"params": [grad_mag_param], "lr": defocus_gradient_magnitude_lr},
                {
                    "params": [grad_angle_u, grad_angle_v],
                    "lr": defocus_gradient_angle_lr,
                },
            ]
        )
    if (
        defocus_0_fixed is not None
        and defocus_0_param is None
        and grad_mag_param is not None
    ):
        param_groups.extend(
            [
                {"params": [grad_mag_param], "lr": defocus_gradient_magnitude_lr},
                {
                    "params": [grad_angle_u, grad_angle_v],
                    "lr": defocus_gradient_angle_lr,
                },
            ]
        )
    if defocus_0_spline_1d is not None:
        assert grad_mag_spline_1d is not None
        assert grad_angle_u_spline_1d is not None
        assert grad_angle_v_spline_1d is not None
        param_groups.extend(
            [
                {"params": defocus_0_spline_1d.parameters(), "lr": defocus_lr},
                {
                    "params": grad_mag_spline_1d.parameters(),
                    "lr": defocus_gradient_magnitude_lr,
                },
                {
                    "params": grad_angle_u_spline_1d.parameters(),
                    "lr": defocus_gradient_angle_lr,
                },
                {
                    "params": grad_angle_v_spline_1d.parameters(),
                    "lr": defocus_gradient_angle_lr,
                },
            ]
        )
    if optimize_astigmatism:
        param_groups.extend(
            [
                {"params": [astigmatism], "lr": astigmatism_lr},
                {"params": [angle_u, angle_v], "lr": astigmatism_angle_lr},
            ]
        )
    if phase_shift_u_grid_model is not None and phase_shift_v_grid_model is not None:
        param_groups.append(
            {"params": phase_shift_u_grid_model.parameters(), "lr": phase_shift_lr}
        )
        param_groups.append(
            {"params": phase_shift_v_grid_model.parameters(), "lr": phase_shift_lr}
        )
    if phase_shift_quad_params is not None:
        param_groups.append(
            {"params": list(phase_shift_quad_params.values()), "lr": phase_shift_lr}
        )
    optimiser = torch.optim.Adam(params=param_groups)

    defocus_models: list[torch.Tensor] = []
    astigmatism_trace = []
    astigmatism_angle_trace = []
    phase_shift_trace_linear = []
    loss_trace = []
    T = patch_power_spectra.shape[0]
    simulated_ctf2s = None

    for _ in range(n_iterations):
        if optimize_astigmatism:
            if (
                torch.isnan(astigmatism).any()
                or torch.isnan(angle_u).any()
                or torch.isnan(angle_v).any()
                or torch.isinf(astigmatism).any()
                or torch.isinf(angle_u).any()
                or torch.isinf(angle_v).any()
            ):
                with torch.no_grad():
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                continue
            astig_clamped = torch.clamp(astigmatism, min=1e-6)
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
        loss_t_list = []
        for t_idx in range(T):
            patch_ps_t = patch_power_spectra[t_idx]
            positions_t = normalised_patch_positions[t_idx]
            t_norm = positions_t[..., 0:1]
            x_norm = positions_t[..., 1]
            y_norm = positions_t[..., 2]
            if defocus_0_spline_1d is not None:
                assert grad_mag_spline_1d is not None
                assert grad_angle_u_spline_1d is not None
                assert grad_angle_v_spline_1d is not None
                defocus_0_t = defocus_0_spline_1d(t_norm).squeeze(-1)
                grad_mag_t = grad_mag_spline_1d(t_norm).squeeze(-1)
                angle_u_t = grad_angle_u_spline_1d(t_norm).squeeze(-1)
                angle_v_t = grad_angle_v_spline_1d(t_norm).squeeze(-1)
            else:
                assert grad_mag_param is not None
                assert grad_angle_u is not None
                assert grad_angle_v is not None
                if defocus_0_param is not None:
                    defocus_0_t = defocus_0_param.expand_as(x_norm)
                else:
                    assert defocus_0_fixed is not None
                    defocus_0_t = torch.full_like(
                        x_norm, defocus_0_fixed, device=device
                    )
                grad_mag_t = grad_mag_param.expand_as(x_norm)
                angle_u_t = grad_angle_u.expand_as(x_norm)
                angle_v_t = grad_angle_v.expand_as(x_norm)
            _eps = 1e-8
            _norm = torch.sqrt(angle_u_t**2 + angle_v_t**2 + _eps)
            _dir_u = angle_u_t / _norm
            _dir_v = angle_v_t / _norm
            angle_rad = torch.atan2(_dir_v, _dir_u)
            projected = (x_norm - 0.5) * torch.cos(angle_rad) + (
                y_norm - 0.5
            ) * torch.sin(angle_rad)
            predicted_defocus_t = defocus_0_t + grad_mag_t * projected

            u_t = v_t = None
            if (
                phase_shift_u_grid_model is not None
                and phase_shift_v_grid_model is not None
            ):
                u_t = phase_shift_u_grid_model(positions_t).squeeze(-1)
                v_t = phase_shift_v_grid_model(positions_t).squeeze(-1)
                phase_shift_t = torch.remainder(
                    0.5 * torch.atan2(v_t, u_t) * (180.0 / math.pi), 180.0
                )
            elif phase_shift_quad_params is not None:
                x = 2.0 * x_norm - 1.0
                y = 2.0 * y_norm - 1.0
                alpha = phase_shift_quad_params["alpha"]
                s = x * torch.cos(alpha) + y * torch.sin(alpha)
                phase_shift_t = (
                    phase_shift_quad_params["C"]
                    + phase_shift_quad_params["g"] * s
                    + phase_shift_quad_params["k"] * (s**2)
                )
                phase_shift_t = torch.clamp(phase_shift_t, min=0.0, max=180.0)
            else:
                phase_shift_t = 0

            simulated_ctf2s_t = (
                calculate_ctf_2d(
                    defocus=predicted_defocus_t,
                    voltage=300,
                    spherical_aberration=2.7,
                    amplitude_contrast=0.10,
                    phase_shift=phase_shift_t,
                    pixel_size=pixel_spacing_angstroms,
                    image_shape=image_shape,
                    astigmatism=astig_clamped,
                    astigmatism_angle=astig_angle_clamped,
                    rfft=True,
                    fftshift=False,
                )
                ** 2
            )
            simulated_ctf2s_t = simulated_ctf2s_t * (env_2d**2) * bp_filter
            simulated_ctf2s = simulated_ctf2s_t
            if (
                torch.isnan(simulated_ctf2s_t).any()
                or torch.isinf(simulated_ctf2s_t).any()
            ):
                continue
            model_flat = simulated_ctf2s_t.reshape(-1)
            data_flat = patch_ps_t.reshape(-1)
            eps = 1e-8
            model_norm = (model_flat - model_flat.mean()) / (model_flat.std() + eps)
            data_norm = (data_flat - data_flat.mean()) / (data_flat.std() + eps)
            C_t = (model_norm * data_norm).sum()
            loss_t = -C_t
            if u_t is not None and v_t is not None:
                penalty_t = ((u_t**2 + v_t**2 - 1.0) ** 2).mean()
                loss_t = loss_t + PHASE_SHIFT_UNIT_CIRCLE_PENALTY * penalty_t
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
        if optimize_astigmatism:
            with torch.no_grad():
                if (
                    torch.isnan(astigmatism).any()
                    or torch.isnan(angle_u).any()
                    or torch.isnan(angle_v).any()
                    or torch.isinf(astigmatism).any()
                    or torch.isinf(angle_u).any()
                    or torch.isinf(angle_v).any()
                ):
                    astigmatism.fill_(
                        initial_astigmatism if initial_astigmatism > 0 else 0.05
                    )
                    angle_u.fill_(_angle_u_init)
                    angle_v.fill_(_angle_v_init)
                else:
                    astigmatism.clamp_(min=1e-6)
        if phase_shift_quad_params is not None:
            with torch.no_grad():
                phase_shift_quad_params["C"].clamp_(min=0.0, max=180.0)
        with torch.no_grad():
            if defocus_0_spline_1d is not None:
                assert grad_mag_spline_1d is not None
                assert grad_angle_u_spline_1d is not None
                assert grad_angle_v_spline_1d is not None
                d0 = defocus_0_spline_1d.data.detach().mean().cpu().item()
                gm = grad_mag_spline_1d.data.detach().mean().cpu().item()
                au = grad_angle_u_spline_1d.data.detach().mean().cpu().item()
                av = grad_angle_v_spline_1d.data.detach().mean().cpu().item()
            else:
                assert grad_mag_param is not None
                assert grad_angle_u is not None
                assert grad_angle_v is not None
                d0 = (
                    defocus_0_param.detach().cpu().item()
                    if defocus_0_param is not None
                    else defocus_0_fixed
                )
                gm = grad_mag_param.detach().cpu().item()
                au = grad_angle_u.detach().cpu().item()
                av = grad_angle_v.detach().cpu().item()
            _norm = (au**2 + av**2 + 1e-8) ** 0.5
            angle_deg = (
                math.atan2(av / _norm, au / _norm) * (180.0 / math.pi) + 180.0
            ) % 180.0
        defocus_models.append(torch.tensor([d0, gm, angle_deg, 0.0], device=device))
        if optimize_astigmatism:
            astigmatism_trace.append(float(astigmatism.detach().cpu().item()))
            _norm = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
            _a_rad = torch.atan2(angle_v.detach() / _norm, angle_u.detach() / _norm)
            _a_deg = float((_a_rad * (180.0 / math.pi) + 180.0) % 180.0)
            astigmatism_angle_trace.append(_astig_angle_to_m90_p90(_a_deg))
        if (
            phase_shift_u_grid_model is not None
            and phase_shift_v_grid_model is not None
        ):
            _mu = phase_shift_u_grid_model.data.detach().cpu().mean().item()
            _mv = phase_shift_v_grid_model.data.detach().cpu().mean().item()
            _p = (0.5 * math.degrees(math.atan2(_mv, _mu))) % 180.0
            phase_shift_trace_linear.append(min(_p, 180.0 - _p))
        elif phase_shift_quad_params is not None:
            _c = float(phase_shift_quad_params["C"].detach().cpu().item())
            phase_shift_trace_linear.append(min(_c, 180.0 - _c))

    final_astigmatism = float(astigmatism.detach().cpu().item())
    _fn = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
    _fa_rad = torch.atan2(angle_v.detach() / _fn, angle_u.detach() / _fn)
    _fa_deg = float((_fa_rad.cpu().item() * (180.0 / math.pi) + 180.0) % 180.0)
    final_astigmatism_angle = _astig_angle_to_m90_p90(_fa_deg)
    final_envelope_B = float(envelope_B.detach().cpu().item())
    if defocus_0_spline_1d is not None:
        mean_defocus = float(defocus_0_spline_1d.data.detach().cpu().mean().item())
    elif defocus_0_param is not None:
        mean_defocus = float(defocus_0_param.detach().cpu().item())
    else:
        assert defocus_0_fixed is not None
        mean_defocus = float(defocus_0_fixed)
    final_defocus_u = mean_defocus + final_astigmatism / 2.0
    final_defocus_v = mean_defocus - final_astigmatism / 2.0

    if defocus_0_spline_1d is not None:
        assert grad_mag_spline_1d is not None
        assert grad_angle_u_spline_1d is not None
        assert grad_angle_v_spline_1d is not None
        _gn = torch.sqrt(
            grad_angle_u_spline_1d.data**2 + grad_angle_v_spline_1d.data**2 + 1e-8
        )
        _grad_angle_deg = (
            torch.atan2(
                grad_angle_v_spline_1d.data / _gn,
                grad_angle_u_spline_1d.data / _gn,
            )
            * (180.0 / math.pi)
            + 180.0
        )
        _grad_angle_deg = _grad_angle_deg % 180.0
        final_grad_angle = float(_grad_angle_deg.mean().cpu().item())
        final_defocus_0 = float(defocus_0_spline_1d.data.mean().cpu().item())
        final_grad_mag = float(grad_mag_spline_1d.data.mean().cpu().item())
    else:
        assert grad_mag_param is not None
        assert grad_angle_u is not None
        assert grad_angle_v is not None
        _gn = torch.sqrt(grad_angle_u.detach() ** 2 + grad_angle_v.detach() ** 2 + 1e-8)
        final_grad_angle = float(
            (
                torch.atan2(
                    grad_angle_v.detach() / _gn,
                    grad_angle_u.detach() / _gn,
                )
                .cpu()
                .item()
                * (180.0 / math.pi)
                + 180.0
            )
            % 180.0
        )
        if defocus_0_param is not None:
            final_defocus_0 = float(defocus_0_param.detach().cpu().item())
        else:
            assert defocus_0_fixed is not None
            final_defocus_0 = float(defocus_0_fixed)
        final_grad_mag = float(grad_mag_param.detach().cpu().item())
    defocus_model_obj = LinearDefocusModel(
        defocus_0=final_defocus_0,
        defocus_gradient_magnitude=final_grad_mag,
        defocus_gradient_angle=final_grad_angle,
        defocus_0_spline_data=(
            defocus_0_spline_1d.data.detach().clone()
            if defocus_0_spline_1d is not None
            else None
        ),
        gradient_magnitude_spline_data=(
            grad_mag_spline_1d.data.detach().clone()
            if grad_mag_spline_1d is not None
            else None
        ),
        angle_u_spline_data=(
            grad_angle_u_spline_1d.data.detach().clone()
            if grad_angle_u_spline_1d is not None
            else None
        ),
        angle_v_spline_data=(
            grad_angle_v_spline_1d.data.detach().clone()
            if grad_angle_v_spline_1d is not None
            else None
        ),
    )
    final_phase_shift_deg_linear = None
    final_phase_shift_model_obj_linear = None
    if phase_shift_u_grid_model is not None and phase_shift_v_grid_model is not None:
        _mu = phase_shift_u_grid_model.data.detach().cpu().mean().item()
        _mv = phase_shift_v_grid_model.data.detach().cpu().mean().item()
        _p = (0.5 * math.degrees(math.atan2(_mv, _mu))) % 180.0
        final_phase_shift_deg_linear = min(_p, 180.0 - _p)
        final_phase_shift_model_obj_linear = (
            phase_shift_u_grid_model,
            phase_shift_v_grid_model,
        )
    elif phase_shift_quad_params is not None:
        _c = float(phase_shift_quad_params["C"].detach().cpu().item())
        final_phase_shift_deg_linear = min(_c, 180.0 - _c)
        final_phase_shift_model_obj_linear = QuadraticPhaseShiftModel(
            C=float(phase_shift_quad_params["C"].detach().cpu().item()),
            g=float(phase_shift_quad_params["g"].detach().cpu().item()),
            k=float(phase_shift_quad_params["k"].detach().cpu().item()),
            alpha_rad=float(phase_shift_quad_params["alpha"].detach().cpu().item()),
        )

    if debug:
        return Defocus2DResults(
            defocus_model_type="linear",
            defocus_model=defocus_model_obj,
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
            loss_trace=loss_trace,
            defocus_u=final_defocus_u,
            defocus_v=final_defocus_v,
            phase_shift_degrees=final_phase_shift_deg_linear,
            phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
            phase_shift_model=final_phase_shift_model_obj_linear
            if optimize_phase_shift
            else None,
            phase_shift_trace=phase_shift_trace_linear
            if optimize_phase_shift
            else None,
        )
    return Defocus2DResults(
        defocus_model_type="linear",
        defocus_model=defocus_model_obj,
        astigmatism=final_astigmatism,
        astigmatism_angle=final_astigmatism_angle,
        envelope_B=final_envelope_B,
        defocus_u=final_defocus_u,
        defocus_v=final_defocus_v,
        phase_shift_degrees=final_phase_shift_deg_linear,
        phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
        phase_shift_model=final_phase_shift_model_obj_linear
        if optimize_phase_shift
        else None,
        phase_shift_trace=phase_shift_trace_linear if optimize_phase_shift else None,
    )


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
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    astigmatism_lr: float = 0.05,
    astigmatism_angle_lr: float = 50.0,
    defocus_model: Literal["grid", "linear"] = "grid",
    initial_defocus_gradient_magnitude: float = 0.0,
    initial_defocus_gradient_angle: float = 0.0,
    defocus_gradient_magnitude_lr: float = 0.05,
    defocus_gradient_angle_lr: float = 50.0,
    fix_defocus_0: Optional[float] = None,
    debug: bool = False,
    optimize_phase_shift: bool = False,
    phase_shift_model: Literal["grid", "quadratic"] = "grid",
    initial_phase_shift: float = 0.0,
    phase_shift_lr: float = 5.0,
) -> Defocus2DResults:
    """
    Estimate defocus in 2D from a power spectrum.

    Optimizes a 2D+t defocus model (grid or linear tilt) and optionally astigmatism
    by maximising the correlation between simulated CTF² and patch power spectra,
    looping over the time/frame dimension with gradient accumulation.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Patch power spectra, shape ``(t, gh, gw, ph, pw)`` (frames, patch grid, freq).
    normalised_patch_positions : torch.Tensor
        Normalised patch positions, shape ``(t, gh, gw, 3)`` in [0, 1].
    defocus_grid_resolution : tuple[int, int, int]
        Resolution ``(nt, nh, nw)``. For grid model all three are used; for linear
        only ``nt`` is used (time knots for cubic spline when t>1).
    frequency_fit_range_angstroms : tuple[float, float]
        ``(low, high)`` frequency fit range in angstroms.
    initial_defocus : float
        Initial defocus in microns.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    initial_astigmatism : float, optional
        Initial astigmatism in microns. Default 0.0.
    initial_astigmatism_angle : float, optional
        Initial astigmatism angle in degrees. Default 0.0.
    optimize_astigmatism : bool, optional
        Whether to optimize astigmatism and angle. Default False.
    initial_envelope_B : float, optional
        Initial B-factor for envelope. Default 0.0.
    n_iterations : int, optional
        Number of optimizer steps. Default 100.
    defocus_lr : float, optional
        Learning rate for the defocus (grid or base defocus_0). Default 0.01.
    astigmatism_lr : float, optional
        Learning rate for the astigmatism magnitude (when ``optimize_astigmatism``).
        Default 0.05.
    astigmatism_angle_lr : float, optional
        Learning rate for the astigmatism angle parameters (when
        ``optimize_astigmatism``). Default 50.0.
    defocus_model : {"grid", "linear"}, optional
        Defocus model: "grid" (3D spline) or "linear" (tilt). Default "grid".
    initial_defocus_gradient_magnitude : float, optional
        Initial defocus gradient magnitude for linear model. Default 0.0.
    initial_defocus_gradient_angle : float, optional
        Initial defocus gradient angle in degrees for linear model. Default 0.0.
    defocus_gradient_magnitude_lr : float, optional
        Learning rate for defocus gradient magnitude (linear). Default 0.05.
    defocus_gradient_angle_lr : float, optional
        Learning rate for defocus gradient angle (linear). Default 50.0.
    fix_defocus_0 : float, optional
        If set (e.g. from 2D fit at 1x1), fix defocus_0 and only optimize
        gradient magnitude and angle in the linear model. Default None.
    debug : bool, optional
        If True, return extra fields (traces, simulated CTF², patch spectra).
        Default False.
    optimize_phase_shift : bool, optional
        Whether to estimate phase shift (0-180 deg) alongside defocus. Default False.
    phase_shift_model : {"grid", "quadratic"}, optional
        "grid" (per-patch) or "quadratic" (directional). Default "grid".
    initial_phase_shift : float, optional
        Initial phase shift in degrees when optimizing. Default 0.0.
    phase_shift_lr : float, optional
        Learning rate for phase shift parameters. Default 5.0.

    Returns
    -------
    Defocus2DResults
        Defocus model, astigmatism, astigmatism angle, envelope B, and optional traces.
    """
    if defocus_model == "grid":
        return estimate_defocus_2d_grid(
            patch_power_spectra=patch_power_spectra,
            normalised_patch_positions=normalised_patch_positions,
            defocus_grid_resolution=defocus_grid_resolution,
            frequency_fit_range_angstroms=frequency_fit_range_angstroms,
            initial_defocus=initial_defocus,
            pixel_spacing_angstroms=pixel_spacing_angstroms,
            initial_astigmatism=initial_astigmatism,
            initial_astigmatism_angle=initial_astigmatism_angle,
            optimize_astigmatism=optimize_astigmatism,
            initial_envelope_B=initial_envelope_B,
            n_iterations=n_iterations,
            defocus_lr=defocus_lr,
            astigmatism_lr=astigmatism_lr,
            astigmatism_angle_lr=astigmatism_angle_lr,
            debug=debug,
            optimize_phase_shift=optimize_phase_shift,
            phase_shift_model=phase_shift_model,
            initial_phase_shift=initial_phase_shift,
            phase_shift_lr=phase_shift_lr,
        )
    return estimate_defocus_2d_linear(
        patch_power_spectra=patch_power_spectra,
        normalised_patch_positions=normalised_patch_positions,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=initial_defocus,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=initial_astigmatism,
        initial_astigmatism_angle=initial_astigmatism_angle,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
        n_iterations=n_iterations,
        defocus_lr=defocus_lr,
        astigmatism_lr=astigmatism_lr,
        astigmatism_angle_lr=astigmatism_angle_lr,
        initial_defocus_gradient_magnitude=initial_defocus_gradient_magnitude,
        initial_defocus_gradient_angle=initial_defocus_gradient_angle,
        defocus_gradient_magnitude_lr=defocus_gradient_magnitude_lr,
        defocus_gradient_angle_lr=defocus_gradient_angle_lr,
        fix_defocus_0=fix_defocus_0,
        debug=debug,
        optimize_phase_shift=optimize_phase_shift,
        phase_shift_model=phase_shift_model,
        initial_phase_shift=initial_phase_shift,
        phase_shift_lr=phase_shift_lr,
    )
