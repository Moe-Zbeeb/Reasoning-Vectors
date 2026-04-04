# GRPO Dataset — math_mixed_grpo.jsonl

12,194 training examples for GRPO. Each row has three fields:
- `question` — the math problem text
- `answer` — canonical answer string (integer, decimal, or tuple) used by the reward function to verify correctness
- `source` — origin tag (`gsm8k`, `competition_math_L1/L2/L3`)

**Composition:**

| Source | Count | % |
|---|---|---|
| GSM8k (socratic, train) | 7,473 | 61.3% |
| competition_math Level 1 | 851 | 7.0% |
| competition_math Level 2 | 1,782 | 14.6% |
| competition_math Level 3 | 2,088 | 17.1% |

**How it was built:**
- GSM8k answers extracted from the `#### N` line at the end of each solution
- competition_math answers extracted from `\boxed{}` in the solution, then normalized: LaTeX fractions converted to decimals, comma-formatted numbers stripped, coordinate tuples kept as `(a,b)`. Level 4–5 excluded (too hard for a 3B model to solve during rollouts). Symbolic/expression answers excluded (not verifiable by exact match).
- Shuffled with seed 42

```bash
python grpo/prepare_mixed.py
```

---

# GRPO Dataset — gsm8k_grpo.jsonl

7,473 examples from `openai/gsm8k` (socratic, train). Fields: `question`, `answer`.

```bash
python grpo/prepare_gsm8k.py
```

---

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
