"""Minimal PyTorch implementation of Group Relative Policy Optimization (GRPO).

Reference:
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)
- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import MathExample
from rewards import compute_rewards


@dataclass
class GRPOConfig:
    """Hyperparameters for minimal GRPO training on a single GPU (e.g. Colab T4)."""

    group_size: int = 4  # G: Number of completions sampled per prompt
    max_prompt_length: int = 256
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    lr: float = 5e-5
    weight_decay: float = 0.01
    beta: float = 0.04  # KL penalty coefficient
    clip_eps: float = 0.2  # PPO-style surrogate clipping epsilon
    mu_iterations: int = 2  # Number of policy gradient updates per rollout batch
    micro_batch_size: int = 2  # Sequences per forward/backward pass to save VRAM
    max_grad_norm: float = 1.0
    num_steps: int = 60
    prompts_per_step: int = 1  # Number of distinct prompts per step (total sequences = prompts_per_step * G)
    log_every: int = 5


def get_per_token_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_completion_tokens: int,
) -> torch.Tensor:
    """Compute per-token log-probabilities for the last `num_completion_tokens` tokens.

    Args:
        model: Causal LM returning `.logits` of shape `(B, L, V)`.
        input_ids: Token IDs of shape `(B, L)` containing `[prompt, completion]`.
        attention_mask: Attention mask of shape `(B, L)`.
        num_completion_tokens: Length `C` of the completion slice at the end of `input_ids`.

    Returns:
        Tensor of shape `(B, C)` with per-token log probabilities.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # Logit at position t-1 predicts token at position t.
    # For completion tokens at indices [L - C, ..., L - 1], we need logits at [L - C - 1, ..., L - 2].
    logits = outputs.logits[:, -(num_completion_tokens + 1) : -1, :]  # (B, C, V)
    target_ids = input_ids[:, -num_completion_tokens:]  # (B, C)

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


def compute_group_advantages(
    rewards: torch.Tensor, group_size: int, eps: float = 1e-4
) -> torch.Tensor:
    """Compute group-normalized advantages: A_{b,g} = (r_{b,g} - mean(r_b)) / (std(r_b) + eps).

    Args:
        rewards: 1D tensor of shape `(B * G,)` containing scalar rewards for each rollout.
        group_size: Number of completions `G` per prompt.
        eps: Numerical stability constant.

    Returns:
        1D tensor of shape `(B * G,)` with normalized group-relative advantages.
    """
    grouped = rewards.view(-1, group_size)  # (B, G)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    advantages = (grouped - mean) / (std + eps)
    return advantages.view(-1)


def compute_grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the clipped GRPO surrogate loss + unbiased per-token KL penalty.

    Args:
        per_token_logps: Current policy log-probs `(N, C)`.
        old_per_token_logps: Rollout policy log-probs `(N, C)`.
        ref_per_token_logps: Reference policy log-probs `(N, C)`.
        advantages: Group-relative advantages `(N,)`.
        completion_mask: Binary mask `(N, C)` with 1 for valid completion tokens and 0 for padding.
        clip_eps: Surrogate clipping parameter.
        beta: KL divergence penalty coefficient.

    Returns:
        Tuple of `(loss, stats_dict)`.
    """
    # Importance sampling ratio: pi_theta / pi_old
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)  # (N, 1)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_token_surrogate = torch.min(surr1, surr2)

    # Unbiased non-negative KL estimator (Schulman): exp(log_ref - log_pi) - (log_ref - log_pi) - 1
    log_ref_ratio = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0

    # Combine surrogate objective and KL penalty: maximize (surrogate - beta * KL)
    per_token_loss = -(per_token_surrogate - beta * per_token_kl)

    # Mask out padding tokens after EOS
    masked_loss = per_token_loss * completion_mask
    total_valid_tokens = completion_mask.sum().clamp(min=1.0)
    loss = masked_loss.sum() / total_valid_tokens

    with torch.no_grad():
        mean_kl = (per_token_kl * completion_mask).sum() / total_valid_tokens
        clip_frac = (
            ((ratio - 1.0).abs() > clip_eps).float() * completion_mask
        ).sum() / total_valid_tokens

    return loss, {
        "loss": loss.item(),
        "kl": mean_kl.item(),
        "clip_frac": clip_frac.item(),
    }


def build_completion_mask(
    completion_ids: torch.Tensor, eos_token_id: Optional[int]
) -> torch.Tensor:
    """Create a mask of shape `(N, C)` that is 1 up to and including the first EOS token, and 0 after."""
    if eos_token_id is None:
        return torch.ones_like(completion_ids, dtype=torch.float32)

    is_eos = completion_ids == eos_token_id
    # Cumulative sum along sequence dimension; tokens strictly before the first EOS have cumsum == 0,
    # the first EOS itself has cumsum == 1 (when shifted).
    eos_cumsum = is_eos.cumsum(dim=1)
    # Include the first EOS token itself, mask out subsequent tokens
    mask = (eos_cumsum - is_eos.long()) == 0
    return mask.to(torch.float32)


@torch.no_grad()
def generate_group_rollouts(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    group_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Sample `group_size` completions for each prompt in `prompts`."""
    was_training = model.training
    model.eval()

    # Left-pad prompts for batched generation
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    ).to(device)
    tokenizer.padding_side = orig_padding_side

    # Repeat each prompt `group_size` times: [p_1, p_1, ..., p_2, p_2, ...]
    prompt_ids = encoded["input_ids"].repeat_interleave(group_size, dim=0)
    prompt_mask = encoded["attention_mask"].repeat_interleave(group_size, dim=0)
    prompt_len = prompt_ids.size(1)

    generated_ids = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    completion_ids = generated_ids[:, prompt_len:]
    completion_mask = build_completion_mask(completion_ids, tokenizer.eos_token_id)
    full_attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)

    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    if was_training:
        model.train()

    return generated_ids, full_attention_mask, completion_mask, completions_text


def compute_logps_in_microbatches(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_completion_tokens: int,
    micro_batch_size: int,
    disable_adapter: bool = False,
) -> torch.Tensor:
    """Compute per-token log-probabilities across micro-batches without tracking gradients."""
    all_logps: List[torch.Tensor] = []
    adapter_ctx = (
        model.disable_adapter()
        if (disable_adapter and hasattr(model, "disable_adapter"))
        else nullcontext()
    )
    with torch.no_grad(), adapter_ctx:
        for i in range(0, input_ids.size(0), micro_batch_size):
            mb_ids = input_ids[i : i + micro_batch_size]
            mb_mask = attention_mask[i : i + micro_batch_size]
            logps = get_per_token_logps(model, mb_ids, mb_mask, num_completion_tokens)
            all_logps.append(logps)
    return torch.cat(all_logps, dim=0)


def train_grpo(
    model: nn.Module,
    tokenizer: Any,
    train_examples: List[MathExample],
    config: GRPOConfig,
    device: torch.device,
    ref_model: Optional[nn.Module] = None,
) -> List[Dict[str, float]]:
    """Run GRPO training loop and return step-by-step training metrics history.

    Memory optimization for Colab GPUs:
    - If `model` is a PEFT/LoRA model and `ref_model` is None, we toggle `model.disable_adapter()`
      to evaluate the frozen reference policy without duplicating base weights in VRAM.
    - If `model` is not a PEFT model and `ref_model` is None, we create a frozen copy when `beta > 0`.
    """
    has_peft_adapter = hasattr(model, "disable_adapter")
    if ref_model is None and not has_peft_adapter and config.beta > 0:
        ref_model = copy.deepcopy(model).eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: List[Dict[str, float]] = []
    num_examples = len(train_examples)
    model.train()

    pbar = tqdm(range(1, config.num_steps + 1), desc="GRPO Training")
    for step in pbar:
        # 1. Sample a mini-batch of prompts
        batch_indices = [
            ((step - 1) * config.prompts_per_step + i) % num_examples
            for i in range(config.prompts_per_step)
        ]
        batch_prompts = [train_examples[idx].prompt for idx in batch_indices]
        batch_answers = [
            train_examples[idx].answer
            for idx in batch_indices
            for _ in range(config.group_size)
        ]

        # 2. Generate G rollouts per prompt using current policy
        full_ids, full_mask, completion_mask, completions_text = generate_group_rollouts(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            group_size=config.group_size,
            max_prompt_length=config.max_prompt_length,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            device=device,
        )
        num_completion_tokens = completion_mask.size(1)

        # 3. Score rollouts with format + accuracy reward functions
        rewards_list, reward_metrics = compute_rewards(completions_text, batch_answers)
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32, device=device)

        # 4. Compute group-relative advantages
        advantages = compute_group_advantages(rewards_tensor, group_size=config.group_size)

        # Skip gradient update if all completions in every group received identical rewards (zero variance)
        if advantages.abs().max().item() < 1e-6:
            step_record = {
                "step": float(step),
                "loss": 0.0,
                "kl": 0.0,
                "reward_mean": reward_metrics["reward_mean"],
                "accuracy_rate": reward_metrics["accuracy_rate"],
                "format_rate": reward_metrics["format_rate"],
            }
            history.append(step_record)
            pbar.set_postfix(
                reward=f"{reward_metrics['reward_mean']:.2f}",
                acc=f"{reward_metrics['accuracy_rate']*100:.0f}%",
                kl="0.000",
            )
            continue

        # 5. Compute old policy log-probs and reference policy log-probs
        model.eval()
        if config.mu_iterations > 1:
            old_per_token_logps = compute_logps_in_microbatches(
                model=model,
                input_ids=full_ids,
                attention_mask=full_mask,
                num_completion_tokens=num_completion_tokens,
                micro_batch_size=config.micro_batch_size,
                disable_adapter=False,
            )
        else:
            old_per_token_logps = None

        if config.beta > 0:
            if has_peft_adapter and ref_model is None:
                ref_per_token_logps = compute_logps_in_microbatches(
                    model=model,
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    num_completion_tokens=num_completion_tokens,
                    micro_batch_size=config.micro_batch_size,
                    disable_adapter=True,
                )
            elif ref_model is not None:
                ref_per_token_logps = compute_logps_in_microbatches(
                    model=ref_model,
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    num_completion_tokens=num_completion_tokens,
                    micro_batch_size=config.micro_batch_size,
                    disable_adapter=False,
                )
            else:
                ref_per_token_logps = torch.zeros(
                    (full_ids.size(0), num_completion_tokens), device=device
                )
        else:
            ref_per_token_logps = torch.zeros(
                (full_ids.size(0), num_completion_tokens), device=device
            )

        # 6. Perform mu policy optimization updates over the rollout batch
        model.train()
        total_sequences = full_ids.size(0)
        num_microbatches = max(1, (total_sequences + config.micro_batch_size - 1) // config.micro_batch_size)
        last_loss = 0.0
        last_kl = 0.0

        for _ in range(config.mu_iterations):
            optimizer.zero_grad()
            iter_loss = 0.0
            iter_kl = 0.0

            for mb_start in range(0, total_sequences, config.micro_batch_size):
                mb_end = mb_start + config.micro_batch_size
                mb_ids = full_ids[mb_start:mb_end]
                mb_mask = full_mask[mb_start:mb_end]
                mb_comp_mask = completion_mask[mb_start:mb_end]
                mb_adv = advantages[mb_start:mb_end]
                mb_ref_logps = ref_per_token_logps[mb_start:mb_end]

                mb_logps = get_per_token_logps(
                    model, mb_ids, mb_mask, num_completion_tokens
                )
                mb_old_logps = (
                    old_per_token_logps[mb_start:mb_end]
                    if old_per_token_logps is not None
                    else mb_logps.detach()
                )

                loss, stats = compute_grpo_loss(
                    per_token_logps=mb_logps,
                    old_per_token_logps=mb_old_logps,
                    ref_per_token_logps=mb_ref_logps,
                    advantages=mb_adv,
                    completion_mask=mb_comp_mask,
                    clip_eps=config.clip_eps,
                    beta=config.beta,
                )
                scaled_loss = loss / num_microbatches
                scaled_loss.backward()

                iter_loss += stats["loss"] / num_microbatches
                iter_kl += stats["kl"] / num_microbatches

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            last_loss = iter_loss
            last_kl = iter_kl

        step_record = {
            "step": float(step),
            "loss": last_loss,
            "kl": last_kl,
            "reward_mean": reward_metrics["reward_mean"],
            "accuracy_rate": reward_metrics["accuracy_rate"],
            "format_rate": reward_metrics["format_rate"],
        }
        history.append(step_record)

        pbar.set_postfix(
            reward=f"{reward_metrics['reward_mean']:.2f}",
            acc=f"{reward_metrics['accuracy_rate']*100:.0f}%",
            kl=f"{last_kl:.3f}",
        )

    return history
