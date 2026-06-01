---
license: mit
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
datasets:
- RabotniKuma/Fast-Math-R1-SFT
language:
- en
pipeline_tag: text-generation
tags:
- deepseek-r1
- qwen
- math
- reasoning
- supervised-fine-tuning
- full-parameter-finetune
- aimo
- long-context
library_name: transformers
---

# DeepSeek-R1-Distill-Qwen-14B Fast-Math-R1 SFT

This model is a full-parameter supervised fine-tune of [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) on [RabotniKuma/Fast-Math-R1-SFT](https://huggingface.co/datasets/RabotniKuma/Fast-Math-R1-SFT).

It is intended for math reasoning experiments where the model should show step-by-step reasoning and place the final answer inside `\boxed{}`.

## Prompt Format

Use this system prompt:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Example chat:

```python
messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "Solve the problem here."},
]
```

## Training Details

| Item | Value |
|---|---:|
| Base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Dataset | `RabotniKuma/Fast-Math-R1-SFT` |
| Training type | Full-parameter SFT |
| GPUs used | 6 x NVIDIA H200 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Effective global batch size | 48 |
| Epochs | 10 |
| Max sequence length | 24,000 tokens |
| Packing | Enabled |
| Learning rate | 1e-5 |
| Scheduler | Cosine |
| Precision | bfloat16 |
| Distributed setup | DeepSpeed ZeRO-3 |

The target recipe was based on the Fast-Math-R1 style training flow from `analokmaus/kaggle-aimo2-fast-math-r1`, adapted for this model and dataset.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zbeeb/deepseek-r1-distill-qwen-14b-fast-math-r1-sft-10ep"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "What is 17 * 23?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

This model is intended for:

- Math reasoning research
- AIMO-style problem solving experiments
- Long-context supervised fine-tuning experiments
- Comparing Fast-Math-R1-style SFT against the base distilled R1 model

## Limitations

This model has not been independently benchmarked in this card. It may produce incorrect reasoning, malformed final answers, or answers that look plausible but are wrong. Validate outputs before using them in any setting where correctness matters.

The training run used 24k-token sequences, but practical inference context length depends on the serving stack, GPU memory, and runtime configuration.

## Source Models and Data

- Base model: [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- Dataset: [RabotniKuma/Fast-Math-R1-SFT](https://huggingface.co/datasets/RabotniKuma/Fast-Math-R1-SFT)

## License

The base model is listed on Hugging Face with an MIT license. The training dataset is listed with an Apache-2.0 license. This model card declares MIT for the uploaded fine-tuned model.
