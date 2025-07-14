import torch
from ..estimate_defocus_1d import Defocus1DResults
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("For plotting please install [plot] extras")
    raise ModuleNotFoundError

def plot_1d_spectrum(
        results1d: Defocus1DResults
):
    