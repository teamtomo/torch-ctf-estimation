"""Contrast transfer function estimation for cryo-EM images in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-ctf-estimation")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_ctf_estimation.estimate_ctf import estimate_ctf

__all__ = [
    "estimate_ctf",
]