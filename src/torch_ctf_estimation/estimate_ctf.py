import einops
import torch

from torch_fourier_rescale import fourier_rescale_2d

from torch_ctf_estimation.estimate_defocus_1d import estimate_defocus_1d
from torch_ctf_estimation.estimate_defocus_2d import estimate_defocus_2d
from torch_ctf_estimation.patch_grid import extract_patch_grid
from torch_ctf_estimation.utils.normalize import normalize_image
from torch_ctf_estimation.utils.estimate_background_2d import estimate_background_2d


def estimate_ctf(
    image: torch.Tensor,  # (t, h, w) or (h, w)
    pixel_spacing_angstroms: float,
    defocus_grid_resolution: tuple[int, int, int],  # (t, h, w)
    frequency_fit_range_angstroms: tuple[float, float],  # (low, high)
    defocus_range_microns: tuple[float, float],  # (low, high)
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast_fraction: float,
    patch_sidelength: int = 512,
    debug: bool = False
):
    # coerce to float
    image = image.float()

    # pack image to (t, h, w)
    image, ps = einops.pack([image], pattern="* h w")
    
    # grab image dimensions
    t, h, w = image.shape

    # normalize images to mean 0 std 1
    image = normalize_image(image)
    # cuton, cutoff = frequency_fit_range_angstroms
    # target_spacing = 0.5 * cutoff
    new_spacing = max(3.0, pixel_spacing_angstroms)
    image, _ = fourier_rescale_2d(
        image=image,
        source_spacing=pixel_spacing_angstroms,
        target_spacing=new_spacing
    )
    # extract grid of 2D patches with 50% overlap
    patches, patch_centers = extract_patch_grid(
        images=image,
        patch_shape=(1, patch_sidelength, patch_sidelength),
        patch_step=(1, patch_sidelength // 2, patch_sidelength // 2)
    )
    patches = einops.rearrange(patches, "t gh gw 1 ph pw -> t gh gw ph pw")

    # calculate power spectra of all patches and mean of all ps
    patch_ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2
    mean_ps = einops.reduce(patch_ps, '... ph pw -> ph pw', reduction='mean')

    # estimate defocus in 1D from mean of power spectra
    result1d = estimate_defocus_1d(
        power_spectrum=mean_ps,
        image_sidelength=patch_sidelength,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast_fraction,
        pixel_spacing_angstroms=new_spacing
    )

    
    # estimate 2D background and subtract prior to 2D defocus estimation
    background_2d = estimate_background_2d(
        power_spectrum=mean_ps,
        image_sidelength=patch_sidelength,
    )
    patch_ps -= background_2d

    # estimate defocus in 2D with gradient based optimisation
    image_dimension_lengths = torch.tensor([t - 1, h - 1, w - 1]).float().to(patch_ps.device)
    normalised_patch_positions = patch_centers / image_dimension_lengths
    result2d = estimate_defocus_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=normalised_patch_positions,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=result1d.ctf_model.defocus_um,
        pixel_spacing_angstroms=new_spacing,
        n_patches_per_batch=40,
    )

    return mean_ps, result1d, result2d