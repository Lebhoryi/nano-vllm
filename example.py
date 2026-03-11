import warnings

warnings.filterwarnings(
    "ignore",
    message=".*does not support bfloat16 compilation natively, skipping.*",
    category=UserWarning,
)

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    path = os.path.expanduser("/toolchain/LLM/Qwen3-0.6B-hf/")
    # path = os.path.expanduser("/toolchain/LLM/Qwen2-0.5B-Instruct-hf")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)
    prompts = [
        "鲁迅为什么要打周树人",
        # "列出 100 以内的素数",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
