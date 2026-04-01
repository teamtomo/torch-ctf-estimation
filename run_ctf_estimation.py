#!/usr/bin/env python3
"""
Example script to run CTF estimation using the new API.

Constants are defined at the top; input models (OpticalParams, CTFFittingParams)
are built from them, then estimate_ctf is called with the image and params.

Run from repo root with the package installed, or:
  PYTHONPATH=src python run_ctf_estimation.py
"""

import torch

from torch_ctf_estimation.estimate_ctf import estimate_ctf
from torch_ctf_estimation.models import CTFFittingParams, OpticalParams

# ---------------------------------------------------------------------------
# OpticalParams (defaults from OpticalParams model)
# ---------------------------------------------------------------------------
PIXEL_SPACING_ANGSTROMS = 1.0
VOLTAGE_KEV = 300.0
SPHERICAL_ABERRATION_MM = 2.7
AMPLITUDE_CONTRAST_FRACTION = 0.07

# ---------------------------------------------------------------------------
# CTFFittingParams (defaults from CTFFittingParams model)
# ---------------------------------------------------------------------------
DEFOCUS_GRID_RESOLUTION = (1, 1, 1)  # (nt, nh, nw)
FREQUENCY_FIT_RANGE_ANGSTROMS = (30.0, 5.0)  # (low, high) 1/A
DEFOCUS_RANGE_MICRONS = (0.0, 2.0)
OPTIMIZE_ASTIGMATISM = True
PATCH_SIDELENGTH = 256
DEFOCUS_MODEL = "grid"  # "grid" or "linear"
OPTIMIZE_PHASE_SHIFT = False
PHASE_SHIFT_MODEL = "grid"  # "grid" or "quadratic"

DEBUG = False
USE_1D_DEFOCUS_FOR_SPATIAL = False
# When True, 1D spatial uses equiphase averaging by default (see CTFFittingParams).
USE_EQUIPHASE_FOR_1D_SPATIAL = True
EQUIPHASE_N_THETA = 64
LINEAR_FIX_DEFOCUS_0_FROM_1X1 = False
REFINE_STEPS_1D = 40
N_ITERATIONS_2D = 100
OPTIMIZE_ENVELOPE_1D = True
B_RANGE_1D = (0.0, 200.0)
B_STEP_1D = 5.0
INITIAL_ENVELOPE_B = None  # float or None
INITIAL_PHASE_SHIFT = 0.0


# ---------------------------------------------------------------------------
# Optional: paths and device (outside input models)
# ---------------------------------------------------------------------------
RESULTS_PATH = "results.json"  # or None to not write JSON
# DEVICE = None  # None -> auto (cuda:0 if available else cpu)


def main() -> None:
    """Main function to estimate CTF."""
    optical_params = OpticalParams(
        pixel_spacing_angstroms=PIXEL_SPACING_ANGSTROMS,
        voltage_kev=VOLTAGE_KEV,
        spherical_aberration_mm=SPHERICAL_ABERRATION_MM,
        amplitude_contrast_fraction=AMPLITUDE_CONTRAST_FRACTION,
    )
    fitting_params = CTFFittingParams(
        defocus_grid_resolution=DEFOCUS_GRID_RESOLUTION,
        frequency_fit_range_angstroms=FREQUENCY_FIT_RANGE_ANGSTROMS,
        defocus_range_microns=DEFOCUS_RANGE_MICRONS,
        patch_sidelength=PATCH_SIDELENGTH,
        debug=DEBUG,
        optimize_astigmatism=OPTIMIZE_ASTIGMATISM,
        defocus_model=DEFOCUS_MODEL,
        use_1d_defocus_for_spatial=USE_1D_DEFOCUS_FOR_SPATIAL,
        use_equiphase_for_1d_spatial=USE_EQUIPHASE_FOR_1D_SPATIAL,
        equiphase_n_theta=EQUIPHASE_N_THETA,
        linear_fix_defocus_0_from_1x1=LINEAR_FIX_DEFOCUS_0_FROM_1X1,
        refine_steps_1d=REFINE_STEPS_1D,
        n_iterations_2d=N_ITERATIONS_2D,
        optimize_envelope_1d=OPTIMIZE_ENVELOPE_1D,
        b_range_1d=B_RANGE_1D,
        b_step_1d=B_STEP_1D,
        initial_envelope_B=INITIAL_ENVELOPE_B,
        optimize_phase_shift=OPTIMIZE_PHASE_SHIFT,
        phase_shift_model=PHASE_SHIFT_MODEL,
        initial_phase_shift=INITIAL_PHASE_SHIFT,
    )

    # Use a synthetic image for demo; replace with your image load
    image = torch.randn(512, 512)

    mean_ps, result1d, result2d = estimate_ctf(
        image,
        optical_params,
        fitting_params,
        laser_params=None,
        device=None,
        results_path=RESULTS_PATH,
    )

    print("CTF estimation complete.")
    print(f"  Defocus model type: {result2d.defocus_model_type}")
    print(f"  Defocus u: {result2d.defocus_u}")
    print(f"  Defocus v: {result2d.defocus_v}")
    if RESULTS_PATH:
        print(f"  Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
