"""Export Llama 3.2 Instruct models to ExecuTorch (.pte) format with XNNPACK INT8.

Pipeline:
1. Load Hugging Face checkpoint safetensors (meta-llama/Llama-3.2-1B-Instruct).
2. Convert parameters to Meta parameter naming and RoPE layout expected by ExecuTorch.
3. Export and dynamically quantize weights (INT8) targeting the XNNPACK backend for ARM Cortex-A CPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

__all__ = ["convert_hf_to_meta", "download_hf", "write_export_config", "export_pte"]

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

#: HF key -> Meta key for Llama.
HF_TO_META = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
}


def _permute(t: torch.Tensor, n_heads: int, head_dim: int, dim: int) -> torch.Tensor:
    """Llama-3 RoPE reordering (identical to torchtune's ``_permute``)."""
    return (
        t.view(n_heads, 2, head_dim // 2, dim).transpose(1, 2).reshape((head_dim * n_heads), dim)
    )


def _load_safetensors(input_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for f in sorted(input_dir.glob("*.safetensors")):
        state.update(load_file(str(f)))
    return state


def convert_hf_to_meta(input_dir: Path, output_file: Path, params_output: Path) -> None:
    """Convert an HF Llama checkpoint to the Meta format ExecuTorch expects."""
    config = json.loads((input_dir / "config.json").read_text())
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    dim = config["hidden_size"]
    head_dim = config.get("head_dim") or dim // num_heads

    state = _load_safetensors(input_dir)
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if "rotary_emb.inv_freq" in key:
            continue
        abstract = key
        for i in range(64):
            abstract = abstract.replace(f".{i}.", ".{}.")
        if abstract not in HF_TO_META:
            continue
        new_key = HF_TO_META[abstract]
        if "{}" in new_key:
            layer = key.split(".")[2]
            new_key = new_key.format(layer)
        if "q_proj" in key:
            value = _permute(value, num_heads, head_dim, dim)
        elif "k_proj" in key:
            value = _permute(value, num_kv_heads, head_dim, dim)
        out[new_key] = value

    if config.get("tie_word_embeddings"):
        out["output.weight"] = out["tok_embeddings.weight"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_file)

    rope_scaling = config.get("rope_scaling")
    params = {
        "dim": config["hidden_size"],
        "hidden_dim": config["intermediate_size"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "n_layers": config["num_hidden_layers"],
        "norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "use_scaled_rope": bool(rope_scaling),
        "rope_scale_factor": rope_scaling["factor"] if rope_scaling else 8,
        "vocab_size": config["vocab_size"],
    }
    params_output.parent.mkdir(parents=True, exist_ok=True)
    params_output.write_text(json.dumps(params, indent=2) + "\n")
    print(f"converted {len(out)} tensors -> {output_file}")


def download_hf(repo_id: str, out_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    # Meta's Llama repositories are gated, so the download needs the account's
    # token.  The token is cached by ``huggingface-cli login``; pass it
    # explicitly rather than relying on an environment variable.
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    token = token_file.read_text().strip() if token_file.exists() else None
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
        token=token,
    )
    return Path(path)


def write_export_config(
    checkpoint: Path,
    params: Path,
    out_yaml: Path,
    quant: str = "8da4w",
    max_seq_length: int = 1024,
) -> Path:
    """Write the hydra config for ``executorch.extension.llm.export.export_llm``.

    The exporter picks its quantisation path from the *checkpoint file name*
    (ExecuTorch's llama implementation looks for "int8" / "8da4w" in the
    path), so the checkpoint is presented under the name that matches the
    requested precision.
    """
    if quant == "spinquant":
        quant_block = "quantization:\n  use_spin_quant: native\n"
    elif quant in ("8da4w", "q8da4w"):
        # dynamic 4-bit weights with int8 dynamic activations, quantised at
        # export time so the artefact really is 4 bit (this is the path that
        # matches the published 1.0 GiB SpinQuant build in size).
        quant_block = "quantization:\n  pt2e_quantize: xnnpack_dynamic_qc4\n"
    elif quant == "int8":
        quant_block = ""
    elif quant in ("dyn8", "xnnpack_dynamic"):
        # int8 weights with int8 dynamic activations, quantised at export time
        quant_block = "quantization:\n  pt2e_quantize: xnnpack_dynamic\n"
    else:
        raise ValueError(f"unknown quantisation {quant!r}")

    text = f"""base:
  model_class: llama3_2
  checkpoint: {checkpoint}
  params: {params}
  metadata: '{{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}}'

model:
  # The pip wheel's runtime does not register the custom SDPA operator
  # (llama::custom_sdpa), so the standard SDPA path is used instead.
  use_sdpa_with_kv_cache: False
  use_kv_cache: True
  dtype_override: fp32

export:
  max_seq_length: {max_seq_length}
  max_context_length: {max_seq_length}

{quant_block}backend:
  xnnpack:
    enabled: True
    extended_ops: True
"""
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(text)
    return out_yaml


def export_pte(config_yaml: Path, out_dir: Path) -> Path:
    """Run the ExecuTorch Llama export; returns the produced ``.pte``.

    The export runs through ``scripts/export_worker.py`` so that the 4-bit
    path can be forced eager (see that file for why).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).resolve().parents[2] / "scripts" / "export_worker.py"
    cmd = [sys.executable, str(worker), "--config", str(config_yaml)]
    env = {"HYDRA_FULL_ERROR": "1"}
    import os

    e = dict(os.environ)
    e.update(env)
    subprocess.run(cmd, check=True, cwd=str(out_dir), env=e)
    ptes = sorted(out_dir.rglob("*.pte"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ptes:
        raise RuntimeError("export produced no .pte")
    return ptes[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--size", default="1b")
    ap.add_argument("--quant", default="spinquant")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    base = MODELS / f"llama32_{args.size}"
    hf_dir = base / "hf"
    if not args.skip_download or not hf_dir.exists():
        print(f"downloading {args.repo} ...")
        download_hf(args.repo, hf_dir)

    ckpt = base / "model_meta.pth"
    params = base / "params.json"
    convert_hf_to_meta(hf_dir, ckpt, params)

    # The exporter selects a *source transform* from the checkpoint file name
    # ("int8" / "8da4w" in the path).  We do not want that for the 4-bit
    # path: the weights are quantised by the graph quantiser at export time
    # instead, so the checkpoint is presented under a neutral name.
    if args.quant in ("8da4w", "q8da4w", "dyn8", "xnnpack_dynamic"):
        ckpt_export = base / "model_bf16.pth"
    elif args.quant == "int8":
        ckpt_export = base / "model_int8.pth"
    else:
        ckpt_export = ckpt
    if ckpt_export != ckpt:
        shutil.copy2(ckpt, ckpt_export)

    cfg = write_export_config(ckpt_export, params, base / "export.yaml", args.quant, args.max_seq_length)
    out_dir = base / f"export_{args.quant}"
    pte = export_pte(cfg, out_dir)
    final = MODELS / f"llama32_{args.size}_{args.quant}.pte"
    shutil.copy2(pte, final)
    print(f"artefact: {final} ({final.stat().st_size / 2**20:.1f} MiB)")
    # keep the tokenizer next to the artefact for the runner
    tok = hf_dir / "tokenizer.json"
    if tok.exists():
        shutil.copy2(tok, final.with_name(final.name + ".tokenizer.json"))


if __name__ == "__main__":
    main()
