# Meta_SecAlign Local Patches

This directory documents the local patch layer required to run `DefensiveToken` models through the official `Meta_SecAlign` evaluation harness.

Applied by `scripts/bootstrap_official_stack.py`:

1. Trusted instruction role is changed from `user` to `system`.
2. Untrusted external data role is changed from `input` to `user`.
3. `tokenizer.apply_chat_template(..., add_defensive_tokens=False)` is injected into the relevant call sites.

These changes are intentionally small and idempotent so the upstream repositories remain the source of truth for benchmark logic and metrics.
