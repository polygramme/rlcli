# rlcli — A CLI Interface for Continual Learning

rlcli runs fused GSPO on your own GPUs in one forward-backward call, the loss hosted Tinker does not serve. Backed by [SkyRL](https://github.com/NovaSky-AI/SkyRL), driven from your terminal.

rlcli is a Python CLI that runs fused RL losses like **GSPO** on your own GPUs in a single `forward_backward` call. Hosted Tinker does not serve GSPO natively: reproducing it there costs a 2-pass round-trip (fetch logprobs, compute the loss client-side, ship the reweighted batch back). rlcli gives you the fused 1-pass path locally, and [our benchmark](benchmarks/last_gspo_bench.json) measures a **23% step-time reduction and 29.8% throughput gain** on Qwen3-4B-Instruct-2507.

It is also the missing front door for the whole stack: the official `tinker` CLI has no `train` verb, and SkyRL has no CLI. rlcli wires them together so you can serve a model, run SFT or RL, and sample from checkpoints without leaving your terminal.

```bash
# serve a Tinker-API training server on your hardware
rlcli serve start --base-model Qwen/Qwen3-4B-Instruct-2507 --backend fsdp --gpus 8

# supervised fine-tune on your own conversations
rlcli train sl --model Qwen/Qwen3-4B-Instruct-2507 --dataset conversations.jsonl

# RL with the fused GSPO loss — one forward_backward call, not the 2-pass custom-loss path
rlcli train rl --model Qwen/Qwen3-4B-Instruct-2507 --loss gspo

# agent RL in sandboxed environments: Docker container + instruction + test script = reward
rlcli train harbor --model Qwen/Qwen3-4B-Instruct-2507 --loss gspo --dataset terminal-bench@2.0

# import your agent's chat dumps and fine-tune on them, all on your hardware
rlcli import prod-traces.jsonl -f openai | rlcli train sl --dataset - --model Qwen/Qwen3-4B-Instruct-2507
```

Docs: [docs.polygramme.com](https://docs.polygramme.com) · Install: `pip install polygramme-rlcli` (the `rlcli` command; PyPI reserves the bare name — not to be confused with `rl-cli`, Runloop's CLI).

## Benchmarks

Measured, with receipts in [`benchmarks/`](benchmarks/):

- **Fused GSPO vs 2-pass** ([json](benchmarks/last_gspo_bench.json)): same A100, same frozen batch — 12.9s vs 16.8s per step (−23%), 586 vs 451 tok/s (+29.8%).
- **GSM8K end-to-end** ([json](benchmarks/row1_gsm8k.json)): Qwen3-4B-Instruct-2507, 25 GSPO steps — 85.29% → 87.11% on the full 1,319-problem test set, for $3.52 of rented A100 time.

## How it works

- `rlcli serve` manages a SkyRL Tinker server in its own uv venv (`~/.rlcli/server-venv`) — required because skyrl caps `tinker<=0.24.1` while the client uses 0.25.0; they meet over HTTP.
- Backends: `jax` (runs anywhere, CPU ok), `fsdp` / `megatron` (Linux + CUDA; serve the full loss set incl. `gspo`, `cispo`, `dppo`, `ppo_critic`).
- `rlcli train` invokes pinned [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipes programmatically. `--loss gspo` on a JAX server fails fast with a clear error.
- `rlcli train harbor` runs Harbor-format tasks (Dockerfile + instruction + test script) on your local Docker daemon — the test verdict is the reward. No cloud sandbox account needed.
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

Trace import from more sources (LangSmith with verdict-based rewards) → PII redaction → on-policy distillation and multi-tenant LoRA → trace-synthesized environments → the continual-learning loop.

Apache-2.0.
