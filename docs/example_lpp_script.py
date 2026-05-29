#!/usr/bin/env python3
"""
Simple script to run estimate_ctf on an image file.
"""

from pathlib import Path



#!/usr/bin/env python3
"""
Simple script to run estimate_ctf on an image file.
"""

from pathlib import Path

import torch
import mrcfile
from torch_ctf_estimation.utils.plotting import plot_1d_spectrum, plot_2d_spectrum, plot_2d_spectrum_images
from torch_ctf_estimation import estimate_ctf
from torch_ctf_estimation.models import CTFFittingParams, LaserParams, OpticalParams

# ============================================================================
# Configuration - Edit these values as needed
# ============================================================================
laser_params = LaserParams(laser_xy_angle_deg=0.0, model_laser=True, dual_laser=True)
IMAGE_PATH = "n26feb11a_00036en.frames_DWS_copy.mrc"  # Path to your image file
PIXEL_SPACING_ANGSTROMS = 0.96
DEFOCUS_GRID_RESOLUTION = (1, 1, 1)  # (t, h, w)
FREQUENCY_FIT_RANGE_ANGSTROMS = (30.0, 4.0)  # (low, high)
DEFOCUS_RANGE_MICRONS = (0.0, 2.0)  # (low, high)
PHASE_SHIFT_RANGE_DEGREES = (50.0, 70.0)
VOLTAGE_KEV = 300.0
SPHERICAL_ABERRATION_MM = 2.7
AMPLITUDE_CONTRAST_FRACTION = 0.07
PATCH_SIDELENGTH = 256
PLOT = False
OUTPUT_PATH = None  # Set to a file path (e.g., "defocus_field.pt") to save results, or None to skip
DEFOCUS_MODEL = "grid"
PHASE_SHIFT_MODEL = "quadratic"

# ============================================================================
# Main script
# ============================================================================

def load_image(image_path: Path):
    """Load image from file. Supports .mrc files."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    suffix = image_path.suffix.lower()
    
    if suffix == '.mrc':
        image = mrcfile.read(image_path)
        image = torch.tensor(image).float()
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Currently only .mrc files are supported.")
    
    return image


def main():
    # Load image
    print(f"Loading image from: {IMAGE_PATH}")
    image = load_image(IMAGE_PATH)
    print(f"Image shape: {image.shape}")
    
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
        phase_shift_range_degrees=PHASE_SHIFT_RANGE_DEGREES,
        patch_sidelength=PATCH_SIDELENGTH,
        debug=True,
        optimize_astigmatism=True,
        defocus_model=DEFOCUS_MODEL,
        use_1d_defocus_for_spatial=False,
        linear_fix_defocus_0_from_1x1=False,
        refine_steps_1d=100,
        n_iterations_2d=200,
        optimize_envelope_1d=True,
        b_range_1d=(0.0, 200.0),
        b_step_1d=5.0,
        initial_envelope_B=None,
        optimize_phase_shift=True,
        phase_shift_model=PHASE_SHIFT_MODEL,
        mask_laser_axis=True,
        laser_axis_mask_width=0.1,
    )

    # Run CTF estimation
    print("Running CTF estimation...")
    mean_ps, result1d, result2d = estimate_ctf(
        image,
        optical_params,
        fitting_params,
        laser_params=laser_params,
        device=None,
        results_path=OUTPUT_PATH,
    )
    
    print(f"\nCTF estimation complete!")
    
    combined_powerspectrum = torch.hstack(
        [torch.flip(mean_ps[:,2:],dims=(1,0)), mean_ps]
    )
    plot_1d_spectrum(result1d)
    plot_2d_spectrum(result2d, result1d)
    plot_2d_spectrum_images(result2d)
    print(result2d.defocus_model.data)
    

    # Print all attributes of result2d
    print("\nAll attributes of result2d:")
    print("=" * 60)
    for attr in dir(result2d):
        if not attr.startswith('_') and attr != "patch_power_spectra":  # Skip private attributes and patch_power_spectra
            try:
                value = getattr(result2d, attr)
                if not callable(value):  # Skip methods, only show attributes
                    print(f"{attr}: {value}")
            except Exception as e:
                print(f"{attr}: <error accessing: {e}>")
    # Print all attributes of result1d
    print("\nAll attributes of result1d:")
    print("=" * 60)
    for attr in dir(result1d):
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(result1d, attr)
                if not callable(value):  # Skip methods, only show attributes
                    print(f"{attr}: {value}")
            except Exception as e:
                print(f"{attr}: <error accessing: {e}>")
    

if __name__ == "__main__":
    main()