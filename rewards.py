"""Reward functions and answer extraction utilities for GSM8K-style math problems."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

THINK_ANSWER_PATTERN = re.compile(
    r"<think>\s*(.+?)\s*</think>\s*<answer>\s*(.+?)\s*</answer>",
    flags=re.DOTALL,
)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.+?)\s*</answer>", flags=re.DOTALL)
GSM8K_HASH_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(text: str) -> Optional[float]:
    """Normalize a numeric string (removing commas/whitespace) to a float."""
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gsm8k_ground_truth(solution_text: str) -> Tuple[str, str]:
    """Split a raw GSM8K solution string (`reasoning #### final_answer`) into (reasoning, answer)."""
    if "####" in solution_text:
        reasoning, answer_part = solution_text.rsplit("####", 1)
        return reasoning.strip(), answer_part.strip().replace(",", "")
    numbers = NUMBER_PATTERN.findall(solution_text)
    final_ans = numbers[-1].replace(",", "") if numbers else ""
    return solution_text.strip(), final_ans


def extract_predicted_answer(completion: str) -> Optional[str]:
    """Extract the predicted numerical answer from a model completion.

    Priority:
    1. Number inside `<answer>...</answer>` tags
    2. Number following `####`
    3. Last number in the completion
    """
    ans_match = ANSWER_TAG_PATTERN.search(completion)
    if ans_match:
        inner = ans_match.group(1).strip()
        nums = NUMBER_PATTERN.findall(inner)
        if nums:
            return nums[-1].replace(",", "")
        return inner

    hash_match = GSM8K_HASH_PATTERN.search(completion)
    if hash_match:
        return hash_match.group(1).replace(",", "")

    nums = NUMBER_PATTERN.findall(completion)
    if nums:
        return nums[-1].replace(",", "")
    return None


def format_reward_fn(completion: str) -> float:
    """Reward the model for adhering to `<think>...</think><answer>...</answer>` structure.

    Returns:
        0.5 if strict `<think>...</think>\\n<answer>...</answer>` format is followed,
        0.2 if partial tags are present (`<think>` and `<answer>`),
        0.0 otherwise.
    """
    if THINK_ANSWER_PATTERN.search(completion):
        return 0.5
    has_think = "<think>" in completion and "</think>" in completion
    has_answer = "<answer>" in completion and "</answer>" in completion
    if has_think and has_answer:
        return 0.2
    return 0.0


def accuracy_reward_fn(
    completion: str, ground_truth: str, require_answer_tag: bool = False
) -> float:
    """Reward the model for producing the exact numerical answer.

    Args:
        completion: Generated text from the model.
        ground_truth: Ground-truth numeric string (e.g., '72').
        require_answer_tag: If True, only extracts answers inside `<answer>...</answer>`
            during RL training to discourage lucky number mentions in reasoning.

    Returns:
        2.0 for a correct answer inside `<answer>...</answer>`,
        1.0 for a correct answer extracted via fallback (when `require_answer_tag=False`),
        0.0 for an incorrect or missing answer.
    """
    gt_val = normalize_number(ground_truth)
    if gt_val is None:
        return 0.0

    ans_match = ANSWER_TAG_PATTERN.search(completion)
    if ans_match:
        nums = NUMBER_PATTERN.findall(ans_match.group(1))
        if nums:
            pred_val = normalize_number(nums[-1])
            if pred_val is not None and abs(pred_val - gt_val) < 1e-5:
                return 2.0

    if require_answer_tag:
        return 0.0

    pred_str = extract_predicted_answer(completion)
    if pred_str is None:
        return 0.0
    pred_val = normalize_number(pred_str)
    if pred_val is not None and abs(pred_val - gt_val) < 1e-5:
        return 1.0
    return 0.0


def compute_rewards(
    completions: List[str], ground_truths: List[str]
) -> Tuple[List[float], Dict[str, float]]:
    """Compute total scalar rewards (accuracy + format) for a batch of completions."""
    total_rewards: List[float] = []
    acc_hits = 0
    fmt_hits = 0

    for comp, gt in zip(completions, ground_truths):
        r_fmt = format_reward_fn(comp)
        r_acc = accuracy_reward_fn(comp, gt, require_answer_tag=False)
        total_rewards.append(r_fmt + r_acc)
        if r_acc > 0.0:
            acc_hits += 1
        if r_fmt >= 0.5:
            fmt_hits += 1

    n = max(len(completions), 1)
    metrics = {
        "reward_mean": sum(total_rewards) / n,
        "accuracy_rate": acc_hits / n,
        "format_rate": fmt_hits / n,
    }
    return total_rewards, metrics
