import einops
import torch


def normalize_image(image: torch.Tensor):
    # grab shape
    t, h, w = image.shape

    # extract central 50 percent for mean/std calculation
    hl, hu = int(0.25 * h), int(0.75 * h)
    wl, wu = int(0.25 * w), int(0.75 * w)
    image_center = image[:, hl:hu, wl:wu]

    # calculate mean and std
    std, mean = torch.std_mean(image_center, dim=(-3, -2, -1))

    # normalize
    image = (image - mean) / std  # (t, h, w)
    return image