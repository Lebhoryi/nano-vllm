import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384  # 单次batch允许的最大token数（引擎侧限制）
    max_num_seqs: int = 512  # 单次batch允许的最大请求数（引擎侧限制）
    max_model_len: int = 4096  # 单条请求允许的最大长度上限（引擎侧限制）
    gpu_memory_utilization: float = 0.9  # 引擎侧GPU内存使用率上限（引擎侧限制）
    tensor_parallel_size: int = 1  # 模型并行数（引擎侧限制）
    enforce_eager: bool = False  # 是否启用eager模式（引擎侧限制）
    hf_config: AutoConfig | None = None  # 模型配置（引擎侧限制）
    eos: int = -1  # 结束符ID（引擎侧限制）
    kvcache_block_size: int = 256  # 单个KV缓存块大小（引擎侧限制）
    num_kvcache_blocks: int = -1  # 总的KV缓存块数（引擎侧限制）

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0  # KV缓存块大小必须是256的倍数
        assert 1 <= self.tensor_parallel_size <= 8  # 模型并行数必须在1到8之间
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)  # 单条请求允许的最大长度上限不能超过模型最大位置嵌入长度
        assert self.max_num_batched_tokens >= self.max_model_len  # 单次batch允许的最大token数不能小于单条请求允许的最大长度上限
