import einops
import numpy as np
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_fourier_filter.ctf import calculate_ctf_2d
from torch_fourier_filter.bandpass import bandpass_filter
from torch_grid_utils.fftfreq_grid import spatial_frequency_to_fftfreq

from pydantic import BaseModel
from typing import Optional

class Defocus2DResults(BaseModel):
    defocus_model: CubicCatmullRomGrid3d
    patch_power_spectra: Optional[torch.Tensor] = None
    model_trace: Optional[list[torch.Tensor]] = None
    simulated_ctf2s: Optional[torch.Tensor] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            CubicCatmullRomGrid3d: lambda v: v.to_dict(),
            torch.Tensor: lambda v: v.tolist(),
        },
    }

def estimate_defocus_2d(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    defocus_grid_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus: float,
    n_patches_per_batch: int,
    pixel_spacing_angstroms: float,
    debug: bool = False,
) -> Defocus2DResults:
    # grab patch sidelength
    patch_sidelength = patch_power_spectra.shape[-2]

    # if only 1 grid point in t, take mean of all patches
    nt, nh, nw = defocus_grid_resolution
    if nt == 1:
        patch_power_spectra = einops.reduce(patch_power_spectra, "t ... -> 1 ...", reduction="mean")

    # Initialise defocus model as 3D grid with defined resolution at initial defocus
    defocus_grid_data = torch.ones(size=defocus_grid_resolution) * initial_defocus
    defocus_model = CubicCatmullRomGrid3d.from_grid_data(defocus_grid_data)

    # bandpass data to fit range
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(1 / low_ang, spacing=pixel_spacing_angstroms)
    high_fftfreq = spatial_frequency_to_fftfreq(1 / high_ang, spacing=pixel_spacing_angstroms)
    filter = bandpass_filter(
        low=low_fftfreq,
        high=high_fftfreq,
        falloff=0,
        image_shape=(patch_sidelength, patch_sidelength),
        rfft=True,
        fftshift=False,
        device=patch_power_spectra.device
    )
    patch_power_spectra *= filter

    # optimise 2d+t defocus model at grid points
    optimiser = torch.optim.Adam(
        params=defocus_model.parameters(),
        lr=0.01,
    )

    defocus_models = []
    for i in range(400):
        # get random subset of patches and their centers
        _, gh, gw = normalised_patch_positions.shape[:3]
        patch_idx = np.random.randint(
            low=(0, 0), high=(gh, gw), size=(n_patches_per_batch, 2)
        )
        idx_gh, idx_gw = einops.rearrange(patch_idx, 'b idx -> idx b')
        subset_patch_ps = patch_power_spectra[:, idx_gh, idx_gw]
        subset_patch_centers = normalised_patch_positions[:, idx_gh, idx_gw]

        # get predicted defocus at patch centers
        predicted_patch_defoci = defocus_model(subset_patch_centers)
        predicted_patch_defoci = einops.rearrange(predicted_patch_defoci, '... 1 -> ...')

        # simulate CTFˆ2 at predicted defocus for each (t, y, x) position
        simulated_ctf2s = calculate_ctf_2d(
            defocus=predicted_patch_defoci,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.10,
            b_factor=0,
            phase_shift=0,
            pixel_size=pixel_spacing_angstroms,
            image_shape=(patch_sidelength, patch_sidelength),
            astigmatism=0,
            astigmatism_angle=0,
            rfft=True,
            fftshift=False,
        ) ** 2  # (t, ph, pw, h, w)
        simulated_ctf2s *= filter

        # zero gradients, calculate loss and backpropagate
        optimiser.zero_grad()
        difference = subset_patch_ps - simulated_ctf2s
        mean_squared_error = torch.mean(difference ** 2)
        mean_squared_error.backward()
        optimiser.step()

        defocus_models.append(defocus_model.data.detach().clone())
    
    if debug:
        return Defocus2DResults(
            defocus_model=defocus_model,
            simulated_ctf2s=simulated_ctf2s,
            patch_power_spectra=patch_power_spectra,
            model_trace=defocus_models
        )
    else:
        return Defocus2DResults(
            defocus_model=defocus_model
        )
