"""Estimate background in 2D from a power spectrum."""

import einops
import torch
import torchvision.transforms.functional as TF
from torch_fourier_filter.dft_utils import rotational_average_dft_2d


def estimate_background_2d(
    power_spectrum: torch.Tensor, image_sidelength: int
) -> torch.Tensor:
    """Estimate background in 2D from a power spectrum.

    Parameters
    ----------
    image_sidelength: int
        Sidelength of the image.
    power_spectrum: torch.Tensor
        Power spectrum of the image.

    Returns
    -------
    bg_estimate_2d: torch.Tensor
        Background estimate in 2D.
    """
    raps_2d, _ = rotational_average_dft_2d(
        dft=power_spectrum,
        image_shape=(image_sidelength, image_sidelength),
        rfft=True,
        fftshifted=False,
        return_1d_average=False,
    )
    raps_2d[0, 0] = 0
    raps_2d = einops.rearrange(raps_2d, "h w -> 1 1 h w")

    # Scale kernel size and sigma with patch size
    # For a reference size of 512, kernel=25, sigma=10
    # Scale proportionally
    reference_size = 256
    kernel_size = int(25 * image_sidelength / reference_size)
    kernel_size = (
        kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    )  # Must be odd
    sigma = 10.0 * image_sidelength / reference_size

    bg_estimate_2d = TF.gaussian_blur(raps_2d, kernel_size=kernel_size, sigma=sigma)
    bg_estimate_2d = einops.rearrange(bg_estimate_2d, "1 1 h w -> h w")
    return bg_estimate_2d
