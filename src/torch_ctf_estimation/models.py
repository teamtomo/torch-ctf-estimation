from pydantic import BaseModel
import torch
from typing import Optional
from torch_cubic_spline_grids import CubicBSplineGrid1d


class CTF(BaseModel):
    defocus_um: torch.Tensor
    voltage_kev: torch.Tensor
    spherical_aberration_mm: torch.Tensor
    amplitude_contrast_fraction: torch.Tensor
    phase_shift_degrees: torch.Tensor

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            torch.Tensor: lambda v: v.tolist(),
        },
    }

class Defocus1DResults(BaseModel):
    frequencies_1d: torch.Tensor
    powerspectrum_1d: torch.Tensor = None
    background_model: Optional[CubicBSplineGrid1d] = None
    test_defoci: Optional[torch.Tensor] = None
    cross_correlations: Optional[torch.Tensor] = None
    ctf_model: CTF
    low_frequency_fit: Optional[float] = None
    high_frequency_fit: Optional[float] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {
            CubicBSplineGrid1d: lambda v: v.to_dict(),
            torch.Tensor: lambda v: v.tolist(),
        },
    }