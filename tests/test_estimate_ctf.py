import torch
import pytest

from torch_ctf_estimation.estimate_ctf import estimate_ctf


def test_estimate_ctf_2d_image():
    """Test estimate_ctf with a 2D image."""
    # Create a synthetic 2D image
    image = torch.randn(1024, 1024)
    
    # Define typical cryo-EM parameters
    pixel_spacing_angstroms = 1.0
    defocus_grid_resolution = (1, 3, 3)  # (t, h, w)
    frequency_fit_range_angstroms = (30.0, 5.0)  # (low, high)
    defocus_range_microns = (0.5, 5.0)  # (low, high)
    voltage_kev = 300.0
    spherical_aberration_mm = 2.7
    amplitude_contrast_fraction = 0.1
    patch_sidelength = 128
    
    # Run estimation
    defocus_field = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        patch_sidelength=patch_sidelength
    )
    
    # Check output shape matches defocus_grid_resolution
    expected_shape = defocus_grid_resolution
    assert defocus_field.shape == expected_shape
    
    # Check defocus values are within reasonable range
    assert torch.all(defocus_field >= defocus_range_microns[0])
    assert torch.all(defocus_field <= defocus_range_microns[1])


def test_estimate_ctf_3d_image():
    """Test estimate_ctf with a 3D image stack."""
    # Create a synthetic 3D image stack
    image = torch.randn(4, 256, 256)
    
    # Define typical cryo-EM parameters
    pixel_spacing_angstroms = 1.5
    defocus_grid_resolution = (4, 2, 2)  # (t, h, w)
    frequency_fit_range_angstroms = (20.0, 4.0)  # (low, high)
    defocus_range_microns = (1.0, 4.0)  # (low, high)
    voltage_kev = 200.0
    spherical_aberration_mm = 2.0
    amplitude_contrast_fraction = 0.07
    patch_sidelength = 64
    
    # Run estimation
    defocus_field = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        patch_sidelength=patch_sidelength
    )
    
    # Check output shape matches defocus_grid_resolution
    expected_shape = defocus_grid_resolution
    assert defocus_field.shape == expected_shape
    
    # Check defocus values are within reasonable range
    assert torch.all(defocus_field >= defocus_range_microns[0])
    assert torch.all(defocus_field <= defocus_range_microns[1])

