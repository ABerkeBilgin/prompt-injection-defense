import numpy as np
import torch

from .base import BaseAttack


def _rand_permute(size, device="cuda", dim=-1):
    return torch.argsort(torch.rand(size, device=device), dim=dim)


class GCGAttack(BaseAttack):
    name = "gcg"

    def __init__(self, config, *args, **kwargs) -> None:
        self._topk = config.topk
        self._num_coords = config.num_coords
        self._mu = config.mu
        super().__init__(config, *args, **kwargs)
        self._momentum = None

    def _get_name_tokens(self) -> list[str]:
        tokens = super()._get_name_tokens()
        tokens.append(f"k{self._topk}")
        return tokens

    def _param_schedule(self):
        return round(self._num_coords[0] + (self._num_coords[1] - self._num_coords[0]) * self._step / self._num_steps)

    @torch.no_grad()
    def _compute_grad(self, eval_input, **kwargs):
        grad = self._model.compute_grad(eval_input, temperature=self._loss_temperature, return_logits=True, **kwargs)
        if self._mu == 0:
            return grad
        if self._momentum is None:
            self._momentum = torch.zeros_like(grad)
        self._momentum.mul_(self._mu).add_(grad)
        return self._momentum

    @torch.no_grad()
    def _sample_updates(self, optim_ids, *args, grad=None, **kwargs):
        _ = args, kwargs
        if grad is None:
            raise ValueError("grad is required for GCG")
        device = grad.device
        num_coords = min(self._param_schedule(), len(optim_ids))
        if self._not_allowed_tokens is not None:
            grad[:, self._not_allowed_tokens.to(device)] = np.inf
        top_indices = (-grad).topk(self._topk, dim=1).indices
        batch_size = int(self._batch_size * 1.25)
        old_token_ids = optim_ids.repeat(batch_size, 1)
        if num_coords == 1:
            new_token_pos = torch.arange(0, len(optim_ids), len(optim_ids) / batch_size, device=device).type(torch.int64)
            rand_idx = _rand_permute((len(optim_ids), self._topk, 1), device=device, dim=1)
            rand_idx = torch.cat([entry[: (new_token_pos == index).sum()] for index, entry in enumerate(rand_idx)], dim=0)
            new_token_val = torch.gather(top_indices[new_token_pos], 1, rand_idx)
            return old_token_ids.scatter(1, new_token_pos.unsqueeze(-1), new_token_val)
        new_token_pos = _rand_permute((batch_size, len(optim_ids)), device=device, dim=1)[:, :num_coords]
        rand_idx = torch.randint(0, self._topk, (batch_size, num_coords, 1), device=device)
        new_token_val = torch.gather(top_indices[new_token_pos], -1, rand_idx)
        new_token_ids = old_token_ids
        for index in range(num_coords):
            new_token_ids.scatter_(1, new_token_pos[:, index].unsqueeze(-1), new_token_val[:, index])
        return new_token_ids
