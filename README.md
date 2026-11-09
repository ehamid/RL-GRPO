# Minimal PyTorch GRPO vs. SFT for GSM8K Math Reasoning

A clean, minimal, from-scratch PyTorch implementation of **Group Relative Policy Optimization (GRPO)** (as introduced in *DeepSeekMath* and *DeepSeek-R1*) compared head-to-head against a **Supervised Fine-Tuning (SFT)** baseline on GSM8K-style math word problems.

Designed to run out-of-the-box on a single **Google Colab GPU (T4 16GB)**.

---

## 1. SFT vs. GRPO: Algorithmic Comparison

### Supervised Fine-Tuning (SFT Baseline)
SFT maximizes the log-likelihood of ground-truth reasoning traces and answers $y = (y_1, \dots, y_T)$ given a prompt $q$:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log \pi_\theta(y_t \mid q, y_{<t})$$

- **Pros**: Fast, stable, only requires a single forward/backward pass per batch; quickly teaches the output format (`<think>...</think><answer>...</answer>`).
- **Cons**: Purely imitates reference demonstrations (*behavior cloning*). It does not learn from its own mistakes, can memorize surface patterns, and suffers from exposure bias at inference time.

### Group Relative Policy Optimization (GRPO)
Instead of training a separate value/critic network (which doubles memory footprint in PPO), **GRPO** samples a group of $G$ completions $\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$ for each question $q$, scores each completion with a verifiable reward function $r_i = R(q, o_i)$, and normalizes rewards within the group to estimate advantages:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}$$

It then optimizes the clipped surrogate objective with an explicit per-token KL divergence penalty against a frozen reference policy $\pi_{\text{ref}}$:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \left[ \min\left( \rho_{i,t} A_i, \, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i \right) - \beta \, \mathbb{D}_{\text{KL}}\left(\pi_\theta \| \pi_{\text{ref}}\right)_{i,t} \right]$$

where:
- $\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ is the per-token importance sampling ratio,
- $\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})_{i,t} = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1$ is Schulman's unbiased non-negative KL estimator.

| Dimension | Supervised Fine-Tuning (SFT) | Group Relative Policy Optimization (GRPO) |
| :--- | :--- | :--- |
| **Supervision Signal** | Requires full ground-truth CoT rationales | Only requires verifiable final answers + format rules |
| **Exploration** | Off-policy (teacher forcing on fixed text) | On-policy (samples $G$ rollouts from its own distribution) |
| **Negative Signal** | None (only pushes up ground-truth tokens) | Explicitly suppresses incorrect reasoning paths ($A_i < 0$) |
| **Critic Network** | None | None (uses group mean/std as baseline) |

---

## 2. Memory Optimizations for Google Colab GPUs

Standard RLHF/PPO requires keeping **4 models** in VRAM simultaneously (Policy, Reference, Critic, Reward Model). This implementation fits comfortably on a single free **Colab T4 (16GB)** by combining four techniques:

1. **No Critic Model**: GRPO replaces the critic with group-normalized rewards.
2. **Rule-Based Verifiable Rewards**: Math accuracy and XML tag formatting are verified deterministically via regex in [`rewards.py`](rewards.py), requiring 0 MB of GPU memory.
3. **Zero-Copy Reference Policy via LoRA Toggle**: When training with LoRA (`peft`), we evaluate $\pi_{\text{ref}}$ by simply wrapping the forward pass in `with model.disable_adapter():` inside [`grpo.py`](grpo.py), avoiding a second copy of the base LLM weights.
4. **Completion-Only Log-Probs & Micro-Batching**: Log-probabilities and softmaxes are sliced strictly over the $C$ completion tokens (`logits[:, -(C+1):-1, :]`) and accumulated across micro-batches.

---

## 3. Repository Structure

- [`grpo.py`](grpo.py) — Core PyTorch GRPO implementation (`get_per_token_logps`, `compute_group_advantages`, `compute_grpo_loss`, `generate_group_rollouts`, `train_grpo`).
- [`sft.py`](sft.py) — Masked-prompt Cross-Entropy SFT baseline trainer (`collate_sft_batch`, `train_sft`).
- [`rewards.py`](rewards.py) — Rule-based GSM8K answer parser, `<think>...</think><answer>...</answer>` format reward, and exact-match accuracy reward.
- [`data.py`](data.py) — GSM8K dataset loader (`openai/gsm8k`) with chat-template formatting and a synthetic word-problem fallback for offline testing.
- [`evaluate.py`](evaluate.py) — Greedy evaluation loop reporting accuracy rate, strict format rate, mean reward, and qualitative samples.
- [`train_and_compare.py`](train_and_compare.py) — End-to-end CLI script that evaluates the Base model, trains & evaluates SFT, trains & evaluates GRPO, and saves comparison plots.
- [`grpo_vs_sft_colab.ipynb`](grpo_vs_sft_colab.ipynb) — Ready-to-run Google Colab notebook.

---

## 4. Quickstart

### Option A: Google Colab
Upload this folder (or open [`grpo_vs_sft_colab.ipynb`](grpo_vs_sft_colab.ipynb) in Colab with a **T4 GPU** runtime) and run all cells.

### Option B: Command Line
```bash
pip install -r requirements.txt

# Run Base vs. SFT vs. GRPO on GSM8K (defaults to Qwen/Qwen2.5-0.5B-Instruct)
python train_and_compare.py \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --train-samples 200 \
    --eval-samples 50 \
    --sft-steps 60 \
    --grpo-steps 60 \
    --group-size 4
```

Outputs are saved to `outputs/comparison_results.png` and `outputs/comparison_metrics.json`.
