# rlcli — A CLI Interface for Continual Learning

rlcli is the open continual-learning loop for agents, on your own GPUs: bring your harness, turn its traces into verifiable sandbox environments, train with RL and self-distillation, and eval every checkpoint in the same sandboxes. Backed by [SkyRL](https://github.com/NovaSky-AI/SkyRL), driven from your terminal. Nothing leaves your machine — traces, environments, training, weights.

The loop, end to end:

1. **Bring your harness.** `rlcli capture` records any OpenAI-compatible agent's calls as traces; `rlcli import` reads OpenAI, Anthropic, LangSmith, Vercel AI SDK and CSV dumps, redacts PII, and joins telemetry scores as rewards.
2. **Traces → environments.** `rlcli synth` drafts a Harbor task per conversation — Dockerfile, instruction, test script — and keeps only tasks an untouched container *fails*, so every task is a reward signal.
3. **Train in sandboxes.** `rlcli train harbor` runs RL against those tasks in local Docker sandboxes with the test verdict as the reward, token-in/token-out across turns, and writes every episode as an ATIF trajectory with its token ids. `rlcli train opsd` self-distills the same weights with a privileged hint; `rlcli train sl` fine-tunes on conversations.
4. **Eval in the sandbox.** `--eval-every` re-runs the held-out split of the same tasks on each checkpoint, so the number you promote on is measured where the agent actually runs.
5. **Serve and go again.** Sample from any checkpoint on your Tinker-API server, capture the next round of traces, repeat.

Underneath, the trainer runs fused RL losses — **GSPO**, DPPO, CISPO — in a single `forward_backward` call, which hosted Tinker does not serve natively (reproducing GSPO there is a 2-pass round-trip: fetch logprobs, compute the loss client-side, ship the reweighted batch back). [Our benchmark](benchmarks/last_gspo_bench.json) measures a **23% step-time reduction and 29.8% throughput gain** on Qwen3-4B-Instruct-2507 from the fused path. rlcli is also the missing front door for the stack: the official `tinker` CLI has no `train` verb and SkyRL has no CLI; rlcli wires them together.

```bash
# serve a Tinker-API training server on your hardware
rlcli serve start --base-model Qwen/Qwen3-4B-Instruct-2507 --backend fsdp --gpus 8

# supervised fine-tune on your own conversations
rlcli train sl --model Qwen/Qwen3-4B-Instruct-2507 --dataset conversations.jsonl

# RL with the fused GSPO loss — one forward_backward call, not the 2-pass custom-loss path
rlcli train rl --model Qwen/Qwen3-4B-Instruct-2507 --loss gspo

# agent RL in sandboxed environments: Docker container + instruction + test script = reward
rlcli train harbor --model Qwen/Qwen3-4B-Instruct-2507 --loss gspo --dataset terminal-bench@2.0

# the same, token-in/token-out, every episode written as an ATIF trajectory with its token ids
rlcli train harbor --model Qwen/Qwen3-4B-Instruct-2507 --loss gspo --dataset ./tasks --trajectories ./episodes

# on-policy self-distillation: the same weights, given a privileged hint, teach the student
rlcli train opsd --model Qwen/Qwen3-4B-Instruct-2507 --dataset prompts.jsonl --teacher-hint "Think step by step and check your arithmetic."

# import your agent's chat dumps and fine-tune on them, all on your hardware
rlcli import prod-traces.jsonl -f openai | rlcli train sl --dataset - --model Qwen/Qwen3-4B-Instruct-2507

# the continual loop: verified traces -> synthesized sandbox tasks -> agent RL
rlcli import runs.jsonl -f langsmith --min-score 0.8 | rlcli synth - --out ./tasks --model gpt-5.2 
rlcli train harbor --model Qwen/Qwen3-4B-Instruct-2507 --dataset ./tasks --loss gspo
```

Docs: [docs.polygramme.com](https://docs.polygramme.com) · Install: `pip install polygramme-rlcli` (the `rlcli` command; PyPI reserves the bare name — not to be confused with `rl-cli`, Runloop's CLI).

## Benchmarks

Measured, with receipts in [`benchmarks/`](benchmarks/):

- **Fused GSPO vs 2-pass** ([json](benchmarks/last_gspo_bench.json)): same A100, same frozen batch — 12.9s vs 16.8s per step (−23%), 586 vs 451 tok/s (+29.8%).

![Fused 1-pass GSPO vs 2-pass custom loss: 23% faster steps, 29.8% higher throughput](benchmarks/bench_gspo_bars.png)

- **GSM8K end-to-end** ([json](benchmarks/row1_gsm8k.json)): Qwen3-4B-Instruct-2507, 25 GSPO steps — 85.29% → 87.11% on the full 1,319-problem test set, for $3.52 of rented A100 time.
- **Cold start on a single L4**: Qwen3-0.6B goes from ~2% to ~60%+ GSM8K train accuracy in 40 GSPO steps.

![Qwen3-0.6B GSM8K train accuracy climbing from ~2% to 60%+ over 40 GSPO steps on one L4](benchmarks/curve_gsm8k_gspo.png)

## How it works

- `rlcli serve` manages a SkyRL Tinker server in its own uv venv (`~/.rlcli/server-venv`) — required because skyrl caps `tinker<=0.24.1` while the client uses 0.25.0; they meet over HTTP.
- Backends: `jax` (runs anywhere, CPU ok), `fsdp` / `megatron` (Linux + CUDA; serve the full loss set incl. `gspo`, `cispo`, `dppo`, `ppo_critic`).
- `rlcli train` invokes pinned [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipes programmatically. `--loss gspo` on a JAX server fails fast with a clear error.
- `rlcli train harbor` runs Harbor-format tasks (Dockerfile + instruction + test script) on your local Docker daemon — the test verdict is the reward. No cloud sandbox account needed.
- `--tito` (implied by `--trajectories`) makes multi-turn rollouts token-in/token-out: each turn extends the previous turn's *sampled* tokens instead of re-rendering the history (`rlcli/tito_bridge.py`, on PrimeIntellect's `renderers`). Without it, chat templates that drop thinking on re-render (Qwen3.5) split every turn into its own datum.
- `--trajectories DIR` records every episode as an [ATIF](https://github.com/laude-institute/harbor/blob/main/rfcs/0001-trajectory-format.md) trajectory — Harbor's interchange format — with prompt and completion token ids inline; see below.
- `rlcli train opsd` runs on-policy distillation from a prompts JSONL (`{"prompt": ...}` or `{"messages": [...]}`): the student samples, a teacher (`--teacher`: any base model or `tinker://` checkpoint on the server; default the student's own base) scores those tokens, and the negative reverse KL becomes the per-token advantage. `--teacher-hint TEXT` gives the teacher privileged context the student never sees.
- `rlcli checkpoint / run / session` pass through to the official tinker CLI, pointed at your server.
- Everything stays in your environment: traces, data, training, weights.

## Dataset format (`train sl`)

One JSON object per line:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
```

`rlcli import` produces this from common chat dumps: `-f openai`, `-f anthropic`
(content blocks incl. tool_use/tool_result), `-f messages` (roles normalized —
`human`→`user`, `ai`→`assistant`), `-f langsmith` (run exports; feedback scores
become a `reward` field, `--min-score` keeps only verified traces; or pull live
with `--project`, needs `pip install "polygramme-rlcli[langsmith]"`), `-f csv`
(one exchange per row), `-f vercel` (AI SDK `parts` messages).

**Tool calls are preserved by default** — assistant `tool_calls` and
`role: "tool"` results survive import in the exact shape the renderers train
on, so your agent's tool use is trainable, not stripped (`--drop-tools` for
text-only). **Telemetry as reward**: `--telemetry events.jsonl` joins product
events (`{"trace_id", "score"}` — thumbs-up, conversion, escalation) onto
conversations by caller-owned trace_id. **PII redaction** before anything is
written: `--redact email --redact api_key` (or `all`), plus custom
`--redact-pattern name=regex` — applied to message content and tool-call
arguments.

**Live capture**: `rlcli capture --upstream https://api.openai.com/v1 --out
traces.jsonl` runs a transparent OpenAI-compatible proxy — point your agent's
base_url at it and every completion (streaming included, tool calls included)
lands in an import-ready trace file.

## Trace → environment synthesis (`synth`)

`rlcli synth` turns imported traces into Harbor-format training environments: an
LLM (any OpenAI-compatible endpoint, including a local vLLM) drafts an
instruction + test script per conversation, rlcli scaffolds the task directory,
and validation builds the Docker image and requires the test to FAIL on an
untouched container — a test an idle agent passes is not a reward signal. Each
task.toml records lineage (`[synth]` source file/line + imported reward), and
the output directory feeds `rlcli train harbor --dataset ./tasks` directly.

## Episodes as ATIF trajectories (`--trajectories`)

`rlcli train harbor --trajectories ./episodes` writes one JSON file per episode
in Harbor's Agent Trajectory Interchange Format (ATIF v1.7), so Harbor's own
viewer and `harbor-atif2otel` read them directly. Each file has the system and
user prompt as steps, one `agent` step per turn with its tool calls, the tool
results it observed, and the episode's reward under `final_metrics.extra`
(`reward`, plus a `rewards` map for named components). Agent steps carry the
token-level record under `metrics`, the slots the spec reserves for it:

```json
{"step_id": 3, "source": "agent", "message": "", "reasoning_content": "I'll list files.",
 "tool_calls": [{"tool_call_id": "call_2", "function_name": "bash", "arguments": {"cmd": "ls"}}],
 "observation": {"results": [{"source_call_id": "call_2", "content": "README.md\nsrc/"}]},
 "metrics": {"prompt_tokens": 285, "completion_tokens": 55,
             "prompt_token_ids": [151644, 8948, "…"], "completion_token_ids": [151667, "…"],
             "extra": {"ckey": "aeebceca830ceb3a"}}}
```

`ckey` is a digest of the first 16 sampled ids; the same key is stamped on the
per-sample capture rows (`rlcli.trace_capture`), which is how a step and the
sequence that produced it are joined without any extra plumbing. Files are
validated against Harbor's pydantic models in the test suite (`pip install
harbor` to run those tests; they skip otherwise). Tool results that answer a
call the model never made keep their content but drop the dangling id, since
ATIF rejects references to unknown calls.

`rlcli.atif.messages_to_trajectory(messages, reward=…)` turns an imported
conversation (`rlcli import` output) into the same shape, so logged
conversations and recorded episodes share one format.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest                       # includes the wire-compat test for extended losses
uv pip install harbor        # optional: validates recorded ATIF files against Harbor's models
uv pip install -e ".[train]" "tinker-cookbook @ git+https://github.com/thinking-machines-lab/tinker-cookbook@f46eddde86e5397138917516a6c69d2ecbf538b1"  # train commands (PyPI forbids the git pin inside the extra)
```

Policy: tinker and tinker-cookbook are pinned dependencies; we do not carry patches against them — compatibility lives in `rlcli/compat.py` and is re-verified by tests on every pin bump.

## Roadmap

Richer environment synthesis (multi-turn tasks, rubric graders, solution replay) → Anthropic-format capture → multi-tenant LoRA → the scheduled continual-learning loop.

Apache-2.0.
