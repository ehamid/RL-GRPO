"""Dataset loading and formatting for GSM8K SFT and GRPO training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rewards import extract_gsm8k_ground_truth

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "First, write your step-by-step reasoning inside <think> and </think> tags. "
    "Then, provide ONLY the final numerical answer inside <answer> and </answer> tags.\n"
    "Example format:\n"
    "<think>\nStep 1: ...\nStep 2: ...\n</think>\n<answer>42</answer>"
)


@dataclass
class MathExample:
    """A single GSM8K-style math problem with formatted prompt, reasoning, and answer."""

    question: str
    reasoning: str
    answer: str
    prompt: str
    sft_completion: str


def format_prompt(question: str, tokenizer: Optional[Any] = None) -> str:
    """Format a math question using the tokenizer's chat template (if available) or a clean fallback."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{question.strip()}\n<|assistant|>\n"


def format_sft_completion(reasoning: str, answer: str, eos_token: str = "") -> str:
    """Format ground-truth GSM8K chain-of-thought into `<think>...</think>\\n<answer>...</answer>`."""
    return f"<think>\n{reasoning.strip()}\n</think>\n<answer>{answer.strip()}</answer>{eos_token}"


def generate_synthetic_gsm8k(num_samples: int = 200, seed: int = 42) -> List[Dict[str, str]]:
    """Generate synthetic GSM8K-style multi-step word problems for fast smoke testing or offline runs."""
    rng = random.Random(seed)
    names = ["Alice", "Bob", "Clara", "David", "Elena", "Frank", "Grace", "Henry"]
    items = ["apples", "books", "marbles", "stickers", "cookies", "pencils"]
    samples: List[Dict[str, str]] = []

    for _ in range(num_samples):
        name = rng.choice(names)
        item = rng.choice(items)
        initial = rng.randint(10, 50)
        bought_packs = rng.randint(2, 6)
        per_pack = rng.randint(4, 12)
        gave_away = rng.randint(3, initial - 2)

        added = bought_packs * per_pack
        subtotal = initial + added
        final_ans = subtotal - gave_away

        question = (
            f"{name} starts with {initial} {item}. {name} then buys {bought_packs} packs "
            f"of {item}, with {per_pack} {item} in each pack. Later, {name} gives {gave_away} "
            f"{item} to a friend. How many {item} does {name} have left?"
        )
        solution = (
            f"{name} buys {bought_packs} * {per_pack} = {added} new {item}. "
            f"Together with the initial {initial} {item}, that makes {initial} + {added} = {subtotal} {item}. "
            f"After giving away {gave_away} {item}, {name} has {subtotal} - {gave_away} = {final_ans} {item} left. "
            f"#### {final_ans}"
        )
        samples.append({"question": question, "answer": solution})

    return samples


def load_gsm8k_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    use_synthetic: bool = False,
    seed: int = 42,
) -> List[MathExample]:
    """Load GSM8K (`openai/gsm8k`) or synthetic fallback and return a list of `MathExample` items."""
    raw_rows: List[Dict[str, str]] = []

    if not use_synthetic:
        try:
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main", split=split)
            if max_samples is not None and max_samples < len(ds):
                ds = ds.shuffle(seed=seed).select(range(max_samples))
            raw_rows = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        except Exception as exc:
            print(f"[data] Falling back to synthetic GSM8K dataset ({exc})")
            raw_rows = generate_synthetic_gsm8k(
                num_samples=max_samples or 200,
                seed=seed + (0 if split == "train" else 999),
            )
    else:
        raw_rows = generate_synthetic_gsm8k(
            num_samples=max_samples or 200,
            seed=seed + (0 if split == "train" else 999),
        )

    eos_token = getattr(tokenizer, "eos_token", "") or ""
    examples: List[MathExample] = []
    for row in raw_rows:
        reasoning, final_answer = extract_gsm8k_ground_truth(row["answer"])
        prompt = format_prompt(row["question"], tokenizer=tokenizer)
        sft_comp = format_sft_completion(reasoning, final_answer, eos_token=eos_token)
        examples.append(
            MathExample(
                question=row["question"],
                reasoning=reasoning,
                answer=final_answer,
                prompt=prompt,
                sft_completion=sft_comp,
            )
        )

    return examples
