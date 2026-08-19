"""rlcli sample — quick generation against a server."""

from __future__ import annotations

import os

import click

DEFAULT_BASE_URL = "http://localhost:8000"


@click.command("sample")
@click.option("--prompt", required=True)
@click.option("--model", "base_model", default=None, help="Base model to sample from.")
@click.option("--checkpoint", "model_path", default=None, help="tinker://... checkpoint path.")
@click.option("--base-url", default=DEFAULT_BASE_URL, show_default=True)
@click.option("--max-tokens", type=int, default=256, show_default=True)
@click.option("--temperature", type=float, default=1.0, show_default=True)
def cli(prompt, base_model, model_path, base_url, max_tokens, temperature):
    """Generate one completion (raw, no chat template)."""
    if not (base_model or model_path):
        raise click.UsageError("Pass --model or --checkpoint.")
    os.environ.setdefault("TINKER_API_KEY", "tml-dummy")
    import tinker
    from tinker import types

    service = tinker.ServiceClient(base_url=base_url)
    sampler = service.create_sampling_client(base_model=base_model, model_path=model_path)

    from transformers import AutoTokenizer

    tok_name = base_model or os.environ.get("RLCLI_TOKENIZER", "")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model_input = types.ModelInput.from_ints(tokenizer.encode(prompt))
    result = sampler.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature),
    ).result()
    tokens = result.sequences[0].tokens
    click.echo(tokenizer.decode(tokens))
