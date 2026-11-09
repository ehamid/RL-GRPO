"""Evaluation utilities for comparing Base, SFT, and GRPO models on GSM8K."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data import MathExample
from rewards import compute_rewards, extract_predicted_answer


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    eval_examples: List[MathExample],
    device: torch.device,
    batch_size: int = 8,
    max_prompt_length: int = 256,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """Evaluate a causal LM on GSM8K examples using greedy decoding."""
    was_training = model.training
    model.eval()

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    all_completions: List[str] = []
    all_ground_truths: List[str] = []
    samples: List[Dict[str, str]] = []

    for i in tqdm(range(0, len(eval_examples), batch_size), desc="Evaluating", leave=False):
        batch = eval_examples[i : i + batch_size]
        prompts = [ex.prompt for ex in batch]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)

        prompt_len = encoded["input_ids"].size(1)
        generated_ids = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        comp_ids = generated_ids[:, prompt_len:]
        completions = tokenizer.batch_decode(comp_ids, skip_special_tokens=True)

        for ex, comp in zip(batch, completions):
            all_completions.append(comp)
            all_ground_truths.append(ex.answer)
            if len(samples) < 5:
                samples.append(
                    {
                        "question": ex.question,
                        "ground_truth": ex.answer,
                        "predicted": extract_predicted_answer(comp) or "N/A",
                        "completion": comp,
                    }
                )

    tokenizer.padding_side = orig_padding_side
    if was_training:
        model.train()

    _, metrics = compute_rewards(all_completions, all_ground_truths)
    return {
        "accuracy_rate": metrics["accuracy_rate"],
        "format_rate": metrics["format_rate"],
        "reward_mean": metrics["reward_mean"],
        "samples": samples,
    }
