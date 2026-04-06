from dataclasses import dataclass

import torch


class LengthMismatchError(Exception):
    pass


@dataclass
class EvalInput:
    dynamic_input_ids: torch.Tensor | None = None
    optim_slice: slice | None = None
    target_slice: slice | None = None
    loss_slice: slice | None = None
    suffix_ids: torch.Tensor | None = None
    target_ids: torch.Tensor | None = None

    def __post_init__(self):
        if self.dynamic_input_ids is not None and self.dynamic_input_ids.ndim != 1:
            raise ValueError("dynamic_input_ids must be 1D")

    def to(self, device: str | torch.device) -> None:
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device, non_blocking=True))