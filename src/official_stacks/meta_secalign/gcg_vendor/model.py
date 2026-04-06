import dataclasses

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class LossOutput:
    losses: torch.Tensor
    logits: torch.Tensor | None = None
    texts: list[str] | None = None
    num_queries: int | None = None
    num_tokens: int | None = None


class TransformersModel:
    def __init__(self, model_name, temperature=0.0, top_p=1.0, max_tokens=512, model=None, tokenizer=None, suffix_manager=None, devices=None, system_message=None, dtype="float32"):
        _ = top_p, system_message, dtype
        _, checkpoint_path = model_name.split("@")
        if devices is None:
            devices = ["cuda"]
        elif isinstance(devices, (int, str, torch.device)):
            devices = [devices]
        self.device = model.device if model is not None else devices[0]
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.suffix_manager = suffix_manager
        self.model = model
        self.tokenizer = tokenizer
        self.embed_layer = self.model.get_input_embeddings()
        self.embed_weights = self.embed_layer.weight.t().detach()
        self.embed_layer.requires_grad_(False)
        self.num_fixed_tokens = 0
        self.model.eval()

    def set_prefix_cache(self, messages):
        _ = messages
        # Disable prefix-cache usage for compatibility with current
        # transformers/Qwen cache APIs used in Colab.
        self.num_fixed_tokens = 0

    def filter_suffixes(self, suffix_ids=None, suffix=None, skipped_suffixes=None):
        _ = suffix
        _, orig_len = suffix_ids.shape
        device = suffix_ids.device
        decoded = self.tokenizer.batch_decode(suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encoded = self.tokenizer(decoded, add_special_tokens=False, return_tensors="pt", padding=True).input_ids.to(device)
        filter_cond = torch.all(encoded[:, :orig_len] == suffix_ids, dim=1)
        if skipped_suffixes is not None:
            is_kept = torch.tensor([item not in skipped_suffixes for item in decoded], device=device, dtype=torch.bool)
        else:
            is_kept = torch.ones(len(decoded), device=device, dtype=torch.bool)
        return filter_cond & is_kept

    def compute_suffix_loss(self, eval_input, batch_size=None, temperature=1.0, max_target_len=None, **kwargs):
        _ = kwargs
        suffix_ids = eval_input.suffix_ids
        dynamic_input_ids = eval_input.dynamic_input_ids
        targets = eval_input.target_ids
        optim_slice = eval_input.optim_slice
        loss_slice = eval_input.loss_slice
        orig_device = suffix_ids.device
        device = self.device
        if max_target_len is not None:
            loss_slice = slice(loss_slice.start, min(loss_slice.stop, loss_slice.start + max_target_len))
            targets = targets[:max_target_len]
            dynamic_input_ids = dynamic_input_ids[: loss_slice.stop + 1]
        num_samples = len(suffix_ids)
        batch_size = min(batch_size or num_samples, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))
        dynamic_input_ids = dynamic_input_ids.to(device)
        batch_dynamic_input_ids = dynamic_input_ids.repeat(batch_size, 1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0).repeat(num_samples, 1)
        loss_list = []
        logits_list = []
        for index in range(num_batches):
            batch_suffix_ids = suffix_ids[index * batch_size : (index + 1) * batch_size].to(device, non_blocking=True)
            batch_targets = targets[index * batch_size : (index + 1) * batch_size].to(device, non_blocking=True)
            current_batch = len(batch_targets)
            batch_dynamic_input_ids[:current_batch, optim_slice] = batch_suffix_ids
            logits, loss, _, loss_slice = self._compute_loss(batch_dynamic_input_ids, batch_targets, loss_slice, num_samples=current_batch, temperature=temperature)
            loss_list.append(loss)
            logits_list.append(logits)
        loss = torch.cat(loss_list, dim=0).to(orig_device, non_blocking=True)
        logits = torch.cat(logits_list, dim=0).to(orig_device, non_blocking=True)
        return LossOutput(losses=loss, logits=logits, num_queries=num_samples)

    def _compute_loss(self, batch_input_ids, batch_targets, loss_slice, num_samples=None, temperature=1.0):
        num_samples = num_samples or len(batch_input_ids)
        input_embeds = self.embed_layer(batch_input_ids)
        logits = self.model(inputs_embeds=input_embeds, use_cache=False).logits[:num_samples]
        logits = logits / temperature
        loss_logits = logits[:, loss_slice]
        loss = F.cross_entropy(loss_logits.permute(0, 2, 1), batch_targets, reduction="none").mean(dim=1)
        return loss_logits, loss, logits, loss_slice

    @torch.no_grad()
    def compute_grad(self, eval_input, temperature=1.0, **kwargs):
        _ = kwargs
        input_ids = eval_input.dynamic_input_ids.to(self.device, non_blocking=True)
        target_ids = eval_input.target_ids.to(self.device, non_blocking=True)
        if target_ids.ndim == 2:
            target_ids.squeeze_(0)
        input_embeds = self.embed_layer(input_ids).unsqueeze(0)
        input_embeds.requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(inputs_embeds=input_embeds, use_cache=False).logits
            loss_logits = logits[:, eval_input.loss_slice].squeeze(0)
            loss = F.cross_entropy(loss_logits / temperature, target_ids)
            embed_grads = torch.autograd.grad(outputs=[loss], inputs=[input_embeds])[0]
        embed_grads = embed_grads[0, eval_input.optim_slice]
        token_grads = embed_grads @ self.embed_weights
        token_grads /= token_grads.norm(dim=-1, keepdim=True)
        return token_grads.to(eval_input.dynamic_input_ids.device, non_blocking=True)
