from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    '''
    序列，保存了 token_ids，以及一些其他信息
    Args:
        token_ids: token_ids 列表
        sampling_params: 采样参数
    
    Notes:
    Sequence 表示“一条生成请求”的运行时状态。LLMEngine.add_request() 创建它，Scheduler 管它的排队和状态切换，BlockManager 给它
    分配 KV cache block，ModelRunner 读取它来构造本轮推理输入。
        1. 保存这条请求当前有哪些 token
        2. 记录它在 KV cache 里占了哪些 block
        3. 在 rank0 和 worker rank 之间做“瘦身版”序列化传输
    '''
    block_size = 256  # 固定写死为 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)  # 序列 ID
        self.status = SequenceStatus.WAITING  # 序列状态
        self.token_ids = copy(token_ids)  # 完整 token_ids 列表，初始化就是 prompt，后面 decode 时会不断 append 新 token
        self.last_token = token_ids[-1]  # 最后一个 token
        self.num_tokens = len(self.token_ids)  # 总 token 数
        self.num_prompt_tokens = len(token_ids)  # 提示词 token 数，方便取前缀，提示词长度
        self.num_cached_tokens = 0  # 已缓存的 token 数，方便计算占用的 KV cache block 数
        self.block_table = []  # 记录它在 KV cache 里占了哪些 block，-1 表示没有 block
        self.temperature = sampling_params.temperature  # 采样温度，方便 ModelRunner 准备采样温度
        self.max_tokens = sampling_params.max_tokens  # 最大 token 数，方便判断是否命中 EOS
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略 EOS，方便判断是否命中 EOS

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        # 生成了多少 tokens
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        # 返回 prompt tokens
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # 生成的 token ids
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        # 已缓存的 KV cache block 数
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        # 总 block 数
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个 block 的 token 数
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        # 获取第 i 个 block 的 token ids
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 添加新 token
        self.token_ids.append(token_id)  # 把新 token 接到 seq.token_ids 尾部
        self.last_token = token_id  # 更新最后一个 token
        self.num_tokens += 1  # 更新总 token 数

    def __getstate__(self):
        '''
        序列化，只保存必要信息，方便传输，跨进程传输。
        Returns:
            tuple: 序列化后的状态
        '''
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        '''
        反序列化，只恢复必要信息，方便传输，跨进程传输。
        Args:
            state: 序列化后的状态
        '''
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
