# rlcli

One-shot CLI for Tinker-API post-training on your own GPUs, backed by [SkyRL](https://github.com/NovaSky-AI/SkyRL).

The [Tinker API](https://tinker-docs.thinkingmachines.ai/) separates training logic from infrastructure. SkyRL implements it on local hardware — with fused RL losses hosted Tinker doesn't serve natively. rlcli is the missing front door: the official `tinker` CLI has no train verb, and SkyRL has no CLI at all.

```bash
# serve a Tinker-API server on your hardware
rlcli serve start --base-model Qwen/Qwen3-4B-Instruct-2507 --backend fsdp --gpus 8

# supervised fine-tune on your own conversations
rlcli train sl --model Qwen/Qwen3-4B-Instruct-2507 --dataset conversations.jsonl

# RL with the fused GSPO loss — one forward_backward call, not Tinker's 2-forward custom-loss path
rlcli train rl --model Qwen/Qwen3-4B-Instruct-2507 --loss gspo

# import your agent's chat dumps and fine-tune on them, all on your hardware
rlcli import prod-traces.jsonl -f openai | rlcli train sl --dataset - --model Qwen/Qwen3-4B-Instruct-2507
```

Not to be confused with `rl-cli` (Runloop's CLI) on PyPI.

## How it works

- `rlcli serve` manages a SkyRL Tinker server in its own uv venv (`~/.rlcli/server-venv`) — required because skyrl caps `tinker<=0.24.1` while the client uses 0.25.0; they meet over HTTP.
- Backends: `jax` (runs anywhere, CPU ok), `fsdp` / `megatron` (Linux + CUDA; serve the full loss set incl. `gspo`, `cispo`, `dppo`, `ppo_critic`).
- `rlcli train` invokes pinned [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipes programmatically. `--loss gspo` on a JAX server fails fast with a clear error.
- `rlcli checkpoint / run / session` pass through to the official tinker CLI, pointed at your server.
- Everything stays in your environment: traces, data, training, weights.

## Dataset format (`train sl`)

One JSON object per line:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
```

`rlcli import` produces this from common chat dumps: `-f openai` (chat-completions
dumps; tool-call turns dropped), `-f anthropic` (Messages API dumps; content blocks
flattened, top-level `system` kept), `-f messages` (already-shaped, roles normalized
— `human`→`user`, `ai`→`assistant`). Imported traces feed SFT/distillation; RL needs
an environment and reward (see roadmap).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                       # includes the wire-compat test for extended losses
uv pip install -e ".[train]" # cookbook (pinned) for the train commands
```

Policy: tinker and tinker-cookbook are pinned dependencies; we do not carry patches against them — compatibility lives in `rlcli/compat.py` and is re-verified by tests on every pin bump.

## Roadmap

Harbor sandboxed agent RL with a local Docker backend → trace import (Claude Code, LangSmith, OpenAI/Anthropic JSONL) with PII redaction → on-policy distillation and multi-tenant LoRA → the continual-learning loop.

Apache-2.0.
