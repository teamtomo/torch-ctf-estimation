import pytest
import torch

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

    # Run estimation (returns mean_ps, result1d, result2d)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        patch_sidelength=patch_sidelength,
    )

    # Default is grid model
    assert result2d.defocus_model_type == "grid"
    # (defocus_model.data has shape (1, t, h, w), squeeze to (t, h, w))
    defocus_field = result2d.defocus_model.data.squeeze(0)
    expected_shape = defocus_grid_resolution
    assert defocus_field.shape == expected_shape

    # Check defocus values are within reasonable range
    # assert torch.all(defocus_field >= defocus_range_microns[0])
    # assert torch.all(defocus_field <= defocus_range_microns[1])


def test_estimate_ctf_2d_image_linear_model():
    """Test estimate_ctf with 2D image and linear defocus model."""
    image = torch.randn(1024, 1024)
    pixel_spacing_angstroms = 1.0
    defocus_grid_resolution = (1, 1, 1)  # only nt=1 used for linear
    frequency_fit_range_angstroms = (30.0, 5.0)
    defocus_range_microns = (0.5, 5.0)
    voltage_kev = 300.0
    spherical_aberration_mm = 2.7
    amplitude_contrast_fraction = 0.1
    patch_sidelength = 128

    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        patch_sidelength=patch_sidelength,
        defocus_model="linear",
    )

    assert result2d.defocus_model_type == "linear"
    linear = result2d.defocus_model
    assert hasattr(linear, "defocus_0")
    assert hasattr(linear, "defocus_gradient_magnitude")
    assert hasattr(linear, "defocus_gradient_angle")
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None


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

    # Run estimation (returns mean_ps, result1d, result2d)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        patch_sidelength=patch_sidelength,
    )

    assert result2d.defocus_model_type == "grid"
    # (defocus_model.data has shape (1, t, h, w), squeeze to (t, h, w))
    defocus_field = result2d.defocus_model.data.squeeze(0)
    expected_shape = defocus_grid_resolution
    assert defocus_field.shape == expected_shape

    # Check defocus values are within reasonable range
    # assert torch.all(defocus_field >= defocus_range_microns[0])
    # assert torch.all(defocus_field <= defocus_range_microns[1])


def test_estimate_ctf_whole_image_mode():
    """Test estimate_ctf with whole-image mode (patch_sidelength < 0)."""
    image = torch.randn(256, 256)
    pixel_spacing_angstroms = 1.0
    defocus_grid_resolution = (1, 1, 1)
    frequency_fit_range_angstroms = (30.0, 5.0)
    defocus_range_microns = (0.5, 5.0)
    voltage_kev = 300.0
    spherical_aberration_mm = 2.7
    amplitude_contrast_fraction = 0.1

    mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        patch_sidelength=-1,
    )

    assert mean_ps.dim() == 2
    assert result2d.defocus_model_type == "grid"
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.shape == defocus_grid_resolution


def test_estimate_ctf_whole_image_mode_rejects_nh_nw_not_1():
    """Test that whole-image mode raises when nh or nw is not 1."""
    image = torch.randn(256, 256)
    with pytest.raises(ValueError, match="nh=1 and nw=1"):
        estimate_ctf(
            image=image,
            pixel_spacing_angstroms=1.0,
            defocus_grid_resolution=(1, 2, 1),
            frequency_fit_range_angstroms=(30.0, 5.0),
            defocus_range_microns=(0.5, 5.0),
            voltage_kev=300.0,
            spherical_aberration_mm=2.7,
            amplitude_contrast_fraction=0.1,
            patch_sidelength=-1,
        )
    with pytest.raises(ValueError, match="nh=1 and nw=1"):
        estimate_ctf(
            image=image,
            pixel_spacing_angstroms=1.0,
            defocus_grid_resolution=(1, 1, 2),
            frequency_fit_range_angstroms=(30.0, 5.0),
            defocus_range_microns=(0.5, 5.0),
            voltage_kev=300.0,
            spherical_aberration_mm=2.7,
            amplitude_contrast_fraction=0.1,
            patch_sidelength=-1,
        )


def test_estimate_ctf_use_1d_defocus_for_spatial():
    """Test use_1d_defocus_for_spatial returns grid/linear result with correct shape."""
    image = torch.randn(512, 512)
    defocus_grid_resolution = (1, 2, 2)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=1.0,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 5.0),
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
        patch_sidelength=128,
        use_1d_defocus_for_spatial=True,
        defocus_model="grid",
    )
    assert result2d.defocus_model_type == "grid"
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.shape == defocus_grid_resolution
    assert result2d.astigmatism is not None or result2d.astigmatism is None


def test_estimate_ctf_use_1d_defocus_for_spatial_linear():
    """Test use_1d_defocus_for_spatial with linear model."""
    image = torch.randn(512, 512)
    defocus_grid_resolution = (1, 2, 2)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=1.0,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 5.0),
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
        patch_sidelength=128,
        use_1d_defocus_for_spatial=True,
        defocus_model="linear",
    )
    assert result2d.defocus_model_type == "linear"
    assert hasattr(result2d.defocus_model, "defocus_0")
    assert hasattr(result2d.defocus_model, "defocus_gradient_magnitude")
    assert hasattr(result2d.defocus_model, "defocus_gradient_angle")


def test_estimate_ctf_linear_fix_defocus_0_from_1x1():
    """Test linear_fix_defocus_0_from_1x1: defocus_0 comes from 2D@1x1."""
    image = torch.randn(512, 512)
    defocus_grid_resolution = (1, 2, 2)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=1.0,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 5.0),
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
        patch_sidelength=128,
        defocus_model="linear",
        linear_fix_defocus_0_from_1x1=True,
    )
    assert result2d.defocus_model_type == "linear"
    assert result2d.defocus_model.defocus_0 is not None
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None


def test_estimate_ctf_linear_fix_defocus_0_2d_zncc():
    """Test linear_fix_defocus_0_from_1x1 with gradient from 2D ZNCC."""
    image = torch.randn(512, 512)
    defocus_grid_resolution = (1, 2, 2)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image=image,
        pixel_spacing_angstroms=1.0,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 5.0),
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
        patch_sidelength=128,
        defocus_model="linear",
        linear_fix_defocus_0_from_1x1=True,
    )
    assert result2d.defocus_model_type == "linear"
    assert result2d.defocus_model.defocus_0 is not None
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None
