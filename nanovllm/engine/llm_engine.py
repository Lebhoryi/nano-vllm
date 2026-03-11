import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)  # 创建配置
        self.ps = []  # 创建进程列表
        self.events = []  # 创建同步事件列表
        ctx = mp.get_context("spawn")  # 创建进程上下文
        for i in range(1, config.tensor_parallel_size):  # 创建其他进程
            event = ctx.Event()  # 创建同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 创建进程
            process.start()  # 启动进程
            self.ps.append(process)  # 添加进程
            self.events.append(event)  # 添加同步事件
        print(".......................")
        self.model_runner = ModelRunner(config, 0, self.events)  # 创建 ModelRunner 0级
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)  # 创建分词器
        config.eos = self.tokenizer.eos_token_id  # 设置结束符ID
        self.scheduler = Scheduler(config)  # 创建调度器，依赖 config.num_kvcache_blocks，由 ModelRunner 分配
        atexit.register(self.exit)  # 注册退出函数

    def exit(self):
        self.model_runner.call("exit")  # 调用 ModelRunner 退出
        del self.model_runner
        for p in self.ps:
            p.join()  # 等待进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)  # 编码提示词
        seq = Sequence(prompt, sampling_params)  # 创建序列，序列中保存的是 token_ids，以及一些其他信息
        self.scheduler.add(seq)  # 添加序列到调度器，只做入队，不做分配

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()  # 调度器调度，分配 KV cache，并返回需要运行的序列列表和是否是 prefill
        token_ids = self.model_runner.call("run", seqs, is_prefill)  # 调用 ModelRunner 运行，返回 token_ids
        self.scheduler.postprocess(seqs, token_ids)  # 调度器后处理，更新序列状态，释放 KV cache
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)  # 计算 token 数量
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()  # 检查是否所有序列都已完成

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)  # 入队
        outputs = {}
        prefill_throughput = decode_throughput = 0.  # 初始化吞吐量
        while not self.is_finished():
            t = perf_counter()  # 记录时间
            output, num_tokens = self.step()  # step 运行，返回输出和 token 数量
            if use_tqdm:
                if num_tokens > 0:  # 计算吞吐量
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)  # 计算吞吐量
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                print(f"seq_id: {seq_id}, output: {self.tokenizer.decode(token_ids)}")
                outputs[seq_id] = token_ids  # 保存输出
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]  # 按照 seq_id 排序输出
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]  # 解码输出
        if use_tqdm:
            pbar.close()
        return outputs
