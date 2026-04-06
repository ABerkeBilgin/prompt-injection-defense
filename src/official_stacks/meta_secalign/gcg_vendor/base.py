import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .eval_input import LengthMismatchError
from .model import TransformersModel


@dataclass
class AttackResult:
    best_loss: float
    best_suffix: str
    num_queries: int
    success: bool


class BaseAttack:
    name = "base"

    def __init__(self, config, model, tokenizer, suffix_manager, not_allowed_tokens, eval_func: Any, **kwargs) -> None:
        _ = kwargs
        self._num_steps = config.num_steps
        self._fixed_params = config.fixed_params
        self._adv_suffix_init = config.adv_suffix_init
        self._init_suffix_len = config.init_suffix_len
        self._batch_size = config.batch_size
        self._mini_batch_size = config.batch_size if config.mini_batch_size <= 0 else config.mini_batch_size
        self._log_freq = config.log_freq
        self._seed = config.seed
        self._seq_len = config.seq_len
        self._loss_temperature = config.loss_temperature
        self._max_queries = config.max_queries
        self._add_space = config.add_space
        self._eval_func = eval_func
        self._skip_mode = config.skip_mode
        self._skip_seen = config.skip_mode == "seen"
        self._skip_visited = self._skip_seen or config.skip_mode == "visited"
        self._model = TransformersModel(
            f"alpaca@{config.log_dir}",
            suffix_manager=suffix_manager,
            model=model,
            tokenizer=tokenizer,
            system_message="",
            max_tokens=100,
            temperature=0.0,
        )
        self._device = self._model.device
        self._not_allowed_tokens = not_allowed_tokens.to(self._device)
        self._tokenizer = tokenizer
        self._suffix_manager = suffix_manager
        self._setup_log_file(config)
        self._start_time = None
        self._step = None
        self._best_loss = None
        self._best_suffix = None
        self._num_queries = 0
        self._seen_suffixes = set()
        self._visited_suffixes = set()

    def _setup_log_file(self, config):
        atk_name = str(self).replace(f"{self.name}_", "")
        log_dir = Path(config.log_dir) / self.name / atk_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / f"{config.sample_id}.jsonl"
        self._log_file.unlink(missing_ok=True)

    def _get_name_tokens(self) -> list[str]:
        init_suffix_len = len(self._adv_suffix_init.split()) if self._init_suffix_len <= 0 else self._init_suffix_len
        tokens = [self.name, f"len{init_suffix_len}", f"{self._num_steps}step", f"bs{self._batch_size}", f"seed{self._seed}", f"l{self._seq_len}", f"t{self._loss_temperature}"]
        if self._fixed_params:
            tokens.append("static")
        return tokens

    def __str__(self):
        return "_".join(self._get_name_tokens())

    def _sample_updates(self, optim_ids, *args, **kwargs):
        raise NotImplementedError

    def _setup_run(self, *, messages, adv_suffix=""):
        self._start_time = time.time()
        self._num_queries = 0
        self._step = None
        self._best_loss, self._best_suffix = float("inf"), adv_suffix
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        if self._fixed_params:
            self._model.set_prefix_cache(messages)

    def _save_best(self, current_loss: float, current_suffix: str) -> None:
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._best_suffix = current_suffix

    def _compute_suffix_loss(self, eval_input):
        output = self._model.compute_suffix_loss(eval_input, batch_size=self._mini_batch_size, temperature=self._loss_temperature)
        self._num_queries += output.num_queries
        return output.losses

    def _compute_grad(self, eval_input, **kwargs):
        raise NotImplementedError

    def _filter_suffixes(self, adv_suffix_ids):
        skipped_suffixes = self._visited_suffixes if self._skip_visited else (self._seen_suffixes if self._skip_seen else None)
        is_valid = self._model.filter_suffixes(suffix_ids=adv_suffix_ids, skipped_suffixes=skipped_suffixes)
        num_valid = is_valid.int().sum().item()
        orig_len = adv_suffix_ids.shape[1]
        batch_size = self._batch_size
        adv_suffix_ids = adv_suffix_ids[is_valid][:batch_size]
        if num_valid < batch_size:
            batch_pad = torch.zeros((batch_size - num_valid, orig_len), dtype=adv_suffix_ids.dtype, device=adv_suffix_ids.device)
            adv_suffix_ids = torch.cat([adv_suffix_ids, batch_pad], dim=0)
        return adv_suffix_ids, min(num_valid, batch_size)

    @torch.no_grad()
    def run(self, messages, target: str) -> AttackResult:
        if self._add_space:
            target = " " + target
        adv_suffix = self._adv_suffix_init
        adv_suffix_ids = self._tokenizer(adv_suffix, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        num_failed = 0
        while True:
            if num_failed >= len(adv_suffix_ids):
                raise RuntimeError("Invalid init suffix.")
            try:
                self._setup_run(messages=messages, adv_suffix=adv_suffix)
            except LengthMismatchError:
                dummy = self._tokenizer("!", add_special_tokens=False).input_ids[0]
                adv_suffix_ids[-num_failed - 1 :] = dummy
                adv_suffix = self._tokenizer.decode(adv_suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                num_failed += 1
                continue
            break
        eval_input = self._suffix_manager.gen_eval_inputs(messages, adv_suffix, target, num_fixed_tokens=self._model.num_fixed_tokens, max_target_len=self._seq_len)
        eval_input.to(self._device)
        optim_slice = eval_input.optim_slice
        passed = True
        for step in range(self._num_steps):
            self._step = step
            dynamic_input_ids = self._suffix_manager.get_input_ids(messages, adv_suffix, target)[0][self._model.num_fixed_tokens :]
            dynamic_input_ids = dynamic_input_ids.to(self._device)
            optim_ids = dynamic_input_ids[optim_slice]
            eval_input.dynamic_input_ids = dynamic_input_ids
            eval_input.suffix_ids = optim_ids
            token_grads = self._compute_grad(eval_input)
            adv_suffix_ids = self._sample_updates(optim_ids=optim_ids, grad=token_grads, optim_slice=optim_slice)
            adv_suffix_ids, num_valid = self._filter_suffixes(adv_suffix_ids)
            adv_suffixes = self._tokenizer.batch_decode(adv_suffix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            self._seen_suffixes.update(adv_suffixes)
            eval_input.suffix_ids = adv_suffix_ids
            losses = self._compute_suffix_loss(eval_input)
            best_index = losses[:num_valid].argmin()
            adv_suffix = adv_suffixes[best_index]
            current_loss = losses[best_index].item()
            self._save_best(current_loss, adv_suffix)
            self._visited_suffixes.add(adv_suffix)
            if (step + 1) % self._log_freq == 0 or step == 0:
                self._num_queries += 1
                result = self._eval_func(adv_suffix, messages)
                passed = result[1] == 0
                self.log({
                    "loss": current_loss,
                    "best_loss": self._best_loss,
                    "success_begin_with": result[1] == 1,
                    "success_in_response": result[0] == 1,
                    "suffix": adv_suffix,
                    "generated": result[2][0][0],
                })
            del token_grads, dynamic_input_ids
            gc.collect()
            if not passed:
                self._best_suffix = adv_suffix
                break
        return AttackResult(best_loss=self._best_loss, best_suffix=self._best_suffix, num_queries=self._num_queries, success=not passed)

    def log(self, log_dict: dict[str, Any]) -> None:
        log_dict["mem"] = 0.0
        log_dict["time_per_step_s"] = (time.time() - self._start_time) / (self._step + 1)
        log_dict["queries"] = self._num_queries
        log_dict["time_min"] = (time.time() - self._start_time) / 60
        log_dict["step"] = self._step
        with self._log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_dict) + "\n")