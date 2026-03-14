"""Models for CTF estimation."""

from typing import Any, Optional

import torch
from pydantic import BaseModel, ConfigDict, field_serializer
from pydantic.functional_serializers import SerializerFunctionWrapHandler
from torch_cubic_spline_grids import CubicBSplineGrid1d


class CTF(BaseModel):
    """CTF model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defocus_um: torch.Tensor
    voltage_kev: torch.Tensor
    spherical_aberration_mm: torch.Tensor
    amplitude_contrast_fraction: torch.Tensor
    phase_shift_degrees: torch.Tensor
    envelope_B: Optional[torch.Tensor] = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return handler(value)


class LinearDefocusModel(BaseModel):
    """Linear (tilt) defocus: defocus_0 + gradient_magnitude * direction in (x,y)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defocus_0: float
    defocus_gradient_magnitude: float
    defocus_gradient_angle: float  # degrees
    # When t>1, optional 1D spline grid data for serialization
    defocus_0_spline_data: Optional[torch.Tensor] = None
    gradient_magnitude_spline_data: Optional[torch.Tensor] = None
    angle_u_spline_data: Optional[torch.Tensor] = None
    angle_v_spline_data: Optional[torch.Tensor] = None

    @property
    def data(self) -> torch.Tensor:
        """
        Summary tensor for compatibility with grid model API (defocus_model.data).

        - Scalar model: shape (3,) [defocus_0, gradient_magnitude, gradient_angle].
        - Spline model (t>1): shape (4, nt) with defocus_0, grad_mag, angle_u, angle_v
          per time knot.
        """
        if self.defocus_0_spline_data is not None:
            return torch.stack(
                [
                    self.defocus_0_spline_data,
                    self.gradient_magnitude_spline_data,
                    self.angle_u_spline_data,
                    self.angle_v_spline_data,
                ]
            )
        return torch.tensor(
            [
                self.defocus_0,
                self.defocus_gradient_magnitude,
                self.defocus_gradient_angle,
            ],
            dtype=torch.float32,
        )

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return handler(value)


class Defocus1DResults(BaseModel):
    """Results from 1D defocus estimation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frequencies_1d: torch.Tensor
    powerspectrum_1d: torch.Tensor = None
    background_model: Optional[CubicBSplineGrid1d] = None
    test_defoci: Optional[torch.Tensor] = None
    cross_correlations: Optional[torch.Tensor] = None
    ctf_model: CTF
    low_frequency_fit: Optional[float] = None
    high_frequency_fit: Optional[float] = None
    envelope_B: Optional[torch.Tensor] = None
    test_B_values: Optional[torch.Tensor] = None
    cross_correlations_2d: Optional[torch.Tensor] = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicBSplineGrid1d):
            return value.to_dict()
        return handler(value)
