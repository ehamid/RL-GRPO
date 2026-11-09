"""End-to-end comparison of Base, SFT, and GRPO on GSM8K-style math problems.

Designed to run comfortably on a single Google Colab T4 (16GB) GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_gsm8k_dataset
from evaluate import evaluate_model
from grpo import GRPOConfig, train_grpo
from sft import SFTConfig, train_sft


def get_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """Select the best available accelerator and floating-point precision."""
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_base_model_and_tokenizer(
    model_name: str, device: torch.device, dtype: torch.dtype
) -> tuple[Any, Any]:
    """Load causal LM and tokenizer configured for left-padded generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.config.use_cache = False
    return model, tokenizer


def attach_fresh_lora(base_model: Any, r: int = 16, alpha: int = 32) -> Any:
    """Wrap the base model with a trainable LoRA adapter."""
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_cfg)


def plot_training_and_comparison(
    sft_history: List[Dict[str, float]],
    grpo_history: List[Dict[str, float]],
    eval_summary: Dict[str, Dict[str, Any]],
    output_path: str = "comparison_results.png",
) -> None:
    """Save a 3-panel plot showing SFT loss, GRPO reward/KL curves, and final accuracy comparison."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping plot generation.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: SFT Training Loss
    sft_steps = [row["step"] for row in sft_history]
    sft_losses = [row["loss"] for row in sft_history]
    axes[0].plot(sft_steps, sft_losses, color="#2563eb", linewidth=2, label="SFT Cross-Entropy Loss")
    axes[0].set_title("SFT Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Panel 2: GRPO Mean Reward & Format Rate
    grpo_steps = [row["step"] for row in grpo_history]
    grpo_rewards = [row["reward_mean"] for row in grpo_history]
    grpo_fmt = [row["format_rate"] for row in grpo_history]
    axes[1].plot(grpo_steps, grpo_rewards, color="#16a34a", linewidth=2, label="Mean Total Reward (max 2.5)")
    axes[1].plot(grpo_steps, grpo_fmt, color="#9333ea", linestyle="--", linewidth=1.8, label="Strict Format Rate")
    axes[1].set_title("GRPO Rollout Reward & Format Adherence")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Score / Rate")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Panel 3: Held-out Test Accuracy & Format Comparison
    models = list(eval_summary.keys())
    accs = [eval_summary[m]["accuracy_rate"] * 100.0 for m in models]
    fmts = [eval_summary[m]["format_rate"] * 100.0 for m in models]

    x = range(len(models))
    width = 0.35
    axes[2].bar([i - width / 2 for i in x], accs, width=width, color="#0ea5e9", label="Accuracy (%)")
    axes[2].bar([i + width / 2 for i in x], fmts, width=width, color="#f59e0b", label="Format Adherence (%)")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(models)
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Test Evaluation: Base vs. SFT vs. GRPO")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved comparison plot to {output_path}")


def print_comparison_table(eval_summary: Dict[str, Dict[str, Any]]) -> None:
    """Print a formatted markdown comparison table of evaluation metrics."""
    print("\n" + "=" * 72)
    print("GSM8K EVALUATION COMPARISON: BASE vs. SFT vs. GRPO")
    print("=" * 72)
    print(f"| {'Model':<14} | {'Accuracy (%)':>14} | {'Format Rate (%)':>16} | {'Mean Reward':>13} |")
    print(f"|{'-' * 16}|{'-' * 16}|{'-' * 18}|{'-' * 15}|")
    for name, res in eval_summary.items():
        acc = res["accuracy_rate"] * 100.0
        fmt = res["format_rate"] * 100.0
        rew = res["reward_mean"]
        print(f"| {name:<14} | {acc:>13.1f}% | {fmt:>15.1f}% | {rew:>13.2f} |")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal PyTorch GRPO vs. SFT on GSM8K-style math problems"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct for fast Colab T4 execution)",
    )
    parser.add_argument("--train-samples", type=int, default=200, help="Number of GSM8K training samples")
    parser.add_argument("--eval-samples", type=int, default=50, help="Number of GSM8K test samples")
    parser.add_argument("--sft-steps", type=int, default=60, help="Number of SFT optimization steps")
    parser.add_argument("--grpo-steps", type=int, default=60, help="Number of GRPO optimization steps")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size G per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for LoRA parameters")
    parser.add_argument("--beta", type=float, default=0.04, help="GRPO KL penalty coefficient")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic GSM8K-style word problems instead of downloading openai/gsm8k",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save metrics JSON and comparison plot",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device, dtype = get_device_and_dtype()
    print(f"[setup] Using device={device}, dtype={dtype}, model={args.model_name}")

    # Load tokenizer first to format dataset prompts with the model's chat template
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = load_gsm8k_dataset(
        split="train",
        max_samples=args.train_samples,
        tokenizer=tokenizer,
        use_synthetic=args.synthetic,
    )
    eval_data = load_gsm8k_dataset(
        split="test",
        max_samples=args.eval_samples,
        tokenizer=tokenizer,
        use_synthetic=args.synthetic,
    )
    print(f"[data] Loaded {len(train_data)} train examples and {len(eval_data)} test examples.")

    eval_summary: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # 1. Base Model Evaluation + SFT Baseline
    # -------------------------------------------------------------------------
    base_model, tokenizer = load_base_model_and_tokenizer(args.model_name, device, dtype)
    print("\n[1/3] Evaluating Base Model...")
    eval_summary["Base"] = evaluate_model(
        base_model,
        tokenizer,
        eval_data,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n[2/3] Training SFT Baseline (LoRA)...")
    sft_model = attach_fresh_lora(base_model)
    sft_cfg = SFTConfig(
        batch_size=2,
        grad_accum_steps=2,
        max_seq_length=256 + args.max_new_tokens,
        lr=args.lr,
        num_steps=args.sft_steps,
    )
    sft_history = train_sft(sft_model, tokenizer, train_data, sft_cfg, device=device)

    print("Evaluating SFT Model...")
    eval_summary["SFT"] = evaluate_model(
        sft_model,
        tokenizer,
        eval_data,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    # Unload SFT adapter to restore the clean base model weights before GRPO
    base_model = sft_model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. GRPO Training & Evaluation
    # -------------------------------------------------------------------------
    print("\n[3/3] Training GRPO Policy (LoRA with Reference-Free Adapter Toggle)...")
    grpo_model = attach_fresh_lora(base_model)
    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        max_prompt_length=256,
        max_new_tokens=args.max_new_tokens,
        lr=args.lr,
        beta=args.beta,
        num_steps=args.grpo_steps,
        prompts_per_step=1,
        micro_batch_size=2,
    )
    grpo_history = train_grpo(grpo_model, tokenizer, train_data, grpo_cfg, device=device)

    print("Evaluating GRPO Model...")
    eval_summary["GRPO"] = evaluate_model(
        grpo_model,
        tokenizer,
        eval_data,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    # -------------------------------------------------------------------------
    # 3. Report & Save Comparison Artifacts
    # -------------------------------------------------------------------------
    print_comparison_table(eval_summary)

    plot_path = str(out_dir / "comparison_results.png")
    plot_training_and_comparison(sft_history, grpo_history, eval_summary, output_path=plot_path)

    results_payload = {
        "config": vars(args),
        "eval_summary": eval_summary,
        "sft_history": sft_history,
        "grpo_history": grpo_history,
    }
    json_path = out_dir / "comparison_metrics.json"
    json_path.write_text(json.dumps(results_payload, indent=2))
    print(f"[save] Wrote detailed metrics and sample generations to {json_path}")


if __name__ == "__main__":
    main()
