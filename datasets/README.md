# GRPO Dataset — math_grpo_10k.jsonl

10k samples for GRPO reinforcement learning, filtered from `open-r1/OpenR1-Math-220K`
(arxiv 2502.17387). Fields: `problem`, `answer`, `source`, `domain`, `llama8b_solve_rate`.

**Difficulty filter:** `llama8b_solve_rate` ∈ [0.2, 0.8] — medium difficulty where RL
signal is most useful (trivial problems give no learning signal; near-impossible ones give
noisy gradients). Deterministic shuffle with seed 42.

Generate with:
```bash
python grpo/prepare_dataset.py
```

HuggingFace source: `open-r1/OpenR1-Math-220K`

---

# SFT Dataset — math_sft_47k.jsonl

~47k samples, CoT only (no Python code), JSONL messages format.

| Source | Count | % |
|--------|-------|---|
| MetaMathQA MATH (AnsAug + Rephrased) | 17,877 | 38.3% |
| MetaMathQA GSM (AnsAug + Rephrased + SV) | 11,970 | 25.7% |
| MathInstruct gsm_rft | 10,561 | 22.6% |
| MathInstruct MATH_train | 4,004 | 8.6% |
| MathInstruct gsm_train | 2,240 | 4.8% |

HuggingFace sources: `meta-math/MetaMathQA`, `TIGER-Lab/MathInstruct`
