"""Pydantic models for CTF estimation inputs (optics, fitting, laser)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class OpticalParams(BaseModel):
    """Optical parameters: pixel spacing, voltage, Cs, amplitude contrast."""

    pixel_spacing_angstroms: float
    voltage_kev: float = 300.0
    spherical_aberration_mm: float = 2.7
    amplitude_contrast_fraction: float = 0.07


class CTFFittingParams(BaseModel):
    """CTF fitting parameters (defocus grid, frequency range, patch size, etc.)."""

    defocus_grid_resolution: tuple[int, int, int]
    frequency_fit_range_angstroms: tuple[float, float]
    defocus_range_microns: tuple[float, float]
    patch_sidelength: int = 256
    debug: bool = False
    optimize_astigmatism: bool = True
    defocus_model: Literal["grid", "linear"] = "grid"
    use_1d_defocus_for_spatial: bool = False
    use_equiphase_for_1d_spatial: bool = True
    equiphase_n_theta: int = 64
    linear_fix_defocus_0_from_1x1: bool = False
    refine_steps_1d: int = 40
    n_iterations_2d: int = 100
    optimize_envelope_1d: bool = True
    b_range_1d: tuple[float, float] = (0.0, 200.0)
    b_step_1d: float = 5.0
    initial_envelope_B: float | None = None
    optimize_phase_shift: bool = False
    phase_shift_model: Literal["grid", "quadratic"] = "grid"
    phase_shift_quadratic_perpendicular_axis: bool = False
    initial_phase_shift: float = 0.0
    mask_laser_axis: bool = False
    laser_axis_mask_width: float = 0.1


class LaserParams(BaseModel):
    """Laser phase plate parameters for optics groups using a laser phase plate.

    Pass this block when you need laser geometry (e.g. axis masking) and/or the
    LPP CTF model. Set ``model_laser=True`` to use the LPP CTF; with
    ``model_laser=False`` a standard CTF is used but ``laser_xy_angle_deg`` and
    ``dual_laser`` still apply when ``mask_laser_axis`` is enabled in
    ``CTFFittingParams``.

    Attributes
    ----------
    model_laser : bool
        If True, use the LPP CTF model for fitting. If False, use the standard
        CTF while still allowing laser-axis masking via ``laser_xy_angle_deg`` and
        ``dual_laser``. Default is False.
    NA : float
        Numerical aperture.
    laser_wavelength_angstrom : float
        Laser wavelength in Angstrom.
    focal_length_angstrom : float
        Focal length in Angstrom.
    laser_xy_angle_deg : float
        Laser angle in the XY plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the XZ plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset in Angstrom.
    laser_trans_offset_angstrom : float
        Transverse offset in Angstrom.
    laser_polarization_angle_deg : float
        Laser polarization angle in degrees.
    peak_phase_deg : float
        Peak phase in degrees.
    dual_laser : bool
        Whether a dual-laser setup is used. Default is False.
    """

    model_laser: bool = False
    NA: float = 0.055
    laser_wavelength_angstrom: float = 10640.0
    focal_length_angstrom: float = 6.8e7
    laser_xy_angle_deg: float = 0.0
    laser_xz_angle_deg: float = 0.0
    laser_long_offset_angstrom: float = 0.0
    laser_trans_offset_angstrom: float = 0.0
    laser_polarization_angle_deg: float = 90.0
    peak_phase_deg: float = 45.0
    dual_laser: bool = True
