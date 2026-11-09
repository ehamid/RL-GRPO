"""Supervised Fine-Tuning (SFT) baseline for GSM8K math problems in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data import MathExample


@dataclass
class SFTConfig:
    """Hyperparameters for SFT baseline training."""

    batch_size: int = 2
    grad_accum_steps: int = 2
    max_seq_length: int = 512
    lr: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_steps: int = 60


def collate_sft_batch(
    examples: List[MathExample],
    tokenizer: Any,
    max_seq_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize `(prompt, sft_completion)` pairs and mask prompt tokens with `-100` in labels."""
    input_ids_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for ex in examples:
        prompt_ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
        comp_ids = tokenizer.encode(ex.sft_completion, add_special_tokens=False)

        full_ids = (prompt_ids + comp_ids)[:max_seq_length]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = ([-100] * prompt_len + comp_ids)[:max_seq_length]

        input_ids_list.append(full_ids)
        labels_list.append(labels)

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_ids: List[List[int]] = []
    padded_mask: List[List[int]] = []
    padded_labels: List[List[int]] = []

    for ids, lbls in zip(input_ids_list, labels_list):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_mask.append([1] * len(ids) + [0] * pad_len)
        padded_labels.append(lbls + [-100] * pad_len)

    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(padded_mask, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
    )


def train_sft(
    model: nn.Module,
    tokenizer: Any,
    train_examples: List[MathExample],
    config: SFTConfig,
    device: torch.device,
) -> List[Dict[str, float]]:
    """Run Supervised Fine-Tuning (SFT) on completion tokens and return step metrics history."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: List[Dict[str, float]] = []
    num_examples = len(train_examples)
    model.train()

    cursor = 0
    pbar = tqdm(range(1, config.num_steps + 1), desc="SFT Training")
    for step in pbar:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(config.grad_accum_steps):
            batch = [
                train_examples[(cursor + i) % num_examples]
                for i in range(config.batch_size)
            ]
            cursor = (cursor + config.batch_size) % num_examples

            input_ids, attention_mask, labels = collate_sft_batch(
                batch, tokenizer, config.max_seq_length, device
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        history.append({"step": float(step), "loss": accum_loss})
        pbar.set_postfix(loss=f"{accum_loss:.4f}")

    return history
