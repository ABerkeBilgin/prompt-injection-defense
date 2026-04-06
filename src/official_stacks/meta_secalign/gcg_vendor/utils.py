from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import torch

from .eval_input import EvalInput


class Role(Enum):
    USER = 1
    ASSISTANT = 2
    SYSTEM = 3


@dataclass
class Message:
    role: Role
    content: str


class SuffixManager:
    def __init__(self, *, tokenizer, use_system_instructions, conv_template):
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        self.sep_tokens = self.tokenizer(self.conv_template.sep, add_special_tokens=False).input_ids
        self.num_tok_sep = len(self.sep_tokens)

    @torch.no_grad()
    def get_input_ids(self, messages, adv_suffix=None, target=None, static_only=False):
        conv = self.conv_template.copy()
        if messages[0].content:
            conv.set_system_message(messages[0].content)
        conv.messages = []
        conv.append_message(conv.roles[0], messages[1].content)

        sep = deepcopy(conv.sep)
        conv.sep = ""
        toks = (
            self.tokenizer(conv.get_prompt(), add_special_tokens=False).input_ids
            + self.tokenizer(" ", add_special_tokens=False).input_ids
            + self.sep_tokens
        )
        num_static_tokens = len(toks) - self.num_tok_sep
        static_input_ids = torch.tensor(toks[:num_static_tokens])
        if static_only:
            conv.sep = sep
            return static_input_ids

        toks = (
            self.tokenizer(conv.get_prompt(), add_special_tokens=False).input_ids
            + self.tokenizer(" ", add_special_tokens=False).input_ids
            + self.tokenizer(adv_suffix, add_special_tokens=False).input_ids
            + self.sep_tokens
        )
        optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep)
        toks = (
            toks
            + self.tokenizer(conv.roles[1], add_special_tokens=False).input_ids
            + self.tokenizer("\n", add_special_tokens=False).input_ids
            + self.tokenizer(target, add_special_tokens=False).input_ids
            + self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False).input_ids
        )
        assistant_role_start = optim_slice.stop
        assistant_role_stop = assistant_role_start + len(self.tokenizer(conv.roles[1], add_special_tokens=False).input_ids) + len(self.tokenizer("\n", add_special_tokens=False).input_ids)
        target_slice = slice(assistant_role_stop, len(toks) - 1)
        loss_slice = slice(assistant_role_stop - 1, len(toks) - 2)
        input_ids = torch.tensor(toks[: target_slice.stop])
        conv.sep = sep
        return input_ids, optim_slice, target_slice, loss_slice

    @torch.no_grad()
    def gen_eval_inputs(self, messages, suffix, target, num_fixed_tokens=0, max_target_len=None):
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
        orig_input_ids, optim_slice, target_slice, loss_slice = self.get_input_ids(messages, suffix, target)
        if max_target_len is not None:
            end = min(target_slice.stop, target_slice.start + max_target_len)
            target_slice = slice(target_slice.start, end)
            loss_slice = slice(loss_slice.start, end - 1)
        orig_input_ids = orig_input_ids[num_fixed_tokens:]
        optim_slice = slice(optim_slice.start - num_fixed_tokens, optim_slice.stop - num_fixed_tokens)
        target_slice = slice(target_slice.start - num_fixed_tokens, target_slice.stop - num_fixed_tokens)
        loss_slice = slice(loss_slice.start - num_fixed_tokens, loss_slice.stop - num_fixed_tokens)
        target_ids = orig_input_ids[target_slice]
        return EvalInput(
            suffix_ids=suffix_ids,
            dynamic_input_ids=orig_input_ids,
            target_ids=target_ids,
            optim_slice=optim_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
        )


def batchify_kv_cache(prefix_cache, batch_size):
    batch_prefix_cache = []
    for key, value in prefix_cache:
        batch_prefix_cache.append((key.repeat(batch_size, 1, 1, 1), value.repeat(batch_size, 1, 1, 1)))
    return batch_prefix_cache


def get_nonascii_toks(tokenizer, device="cpu") -> torch.Tensor:
    def is_ascii(text: str) -> bool:
        return text.isascii() and text.isprintable()

    non_ascii_toks = []
    for index in range(3, tokenizer.vocab_size):
        try:
            token = tokenizer.decode([index], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if not is_ascii(token):
            non_ascii_toks.append(index)
    for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id):
        if token_id is not None:
            non_ascii_toks.append(token_id)
    return torch.tensor(sorted(set(non_ascii_toks)), device=device)


def get_prefix_cache(suffix_manager, model, tokenizer, messages):
    static_input_ids = suffix_manager.get_input_ids(messages, static_only=True)
    device = model.device if hasattr(model, "device") else model.module.device
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(static_input_ids.to(device)).unsqueeze(0)
        outputs = model(inputs_embeds=input_embeds, use_cache=True)
        prefix_cache = outputs.past_key_values
    return prefix_cache, len(static_input_ids)