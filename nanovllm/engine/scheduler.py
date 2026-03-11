from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    '''
    KV block 调度器，负责分配 KV cache，并返回需要运行的序列列表和是否是 prefill
    Args:
        config: 配置，包含最大请求数、最大批量 token 数、结束符 ID、KV cache 块管理器、等待队列、运行队列

    Notes:
        不做模型计算，只做三件事：
        1. 管理请求队列状态
        2. 决定这一轮执行 prefill 还是 decode
        3. 和 BlockManager 配合，分配/回收 KV Cache block
    '''
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs  # 最大请求数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 最大批量 token 数
        self.eos = config.eos  # 结束符 ID
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)  # KV cache 块管理器
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列

    def is_finished(self):
        '''
        检查是否所有序列都已完成
        Returns:
            bool: 是否所有序列都已完成
        '''
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        '''
        添加序列到等待队列
        Args:
            seq: 序列
        '''
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        '''
        调度，分配 KV cache，并返回需要运行的序列列表和是否是 prefill
        Returns:
            tuple[list[Sequence], bool]: 需要运行的序列列表和是否是 prefill
        '''
        # prefill
        scheduled_seqs = [] # 需要运行的序列列表
        num_seqs = 0 # 当前运行的序列数
        num_batched_tokens = 0 # 当前批量 token 数
        while self.waiting and num_seqs < self.max_num_seqs:  # waiting 队列不为空，且当前运行的序列数小于最大请求数
            # waiting 队列转 running 队列，并分配 KV cache
            seq = self.waiting[0] # 获取等待队列中的第一个序列，只看队头，FIFO 队列，先进先出
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break # 如果当前批量 token 数超过最大批量 token 数，或者不能分配 KV cache，则跳出循环
            num_seqs += 1 # 当前运行的序列数加1，num_seqs 表示当前运行的序列数
            self.block_manager.allocate(seq) # 分配 KV cache
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 更新当前批量 token 数，当前轮真正需要处理的 token 数
            seq.status = SequenceStatus.RUNNING # 将序列状态设置为运行中
            self.waiting.popleft() # 从等待队列中移除序列
            self.running.append(seq) # 将序列添加到运行队列，running 队列是 LIFO 队列，后进先出
            scheduled_seqs.append(seq) # 将序列添加到需要运行的序列列表
        if scheduled_seqs:  # prefill 优先级高于 decode
            return scheduled_seqs, True

        # decode，队头优先策略，牺牲队尾，队尾重新排队到队头，假设队列 ABCD
        while self.running and num_seqs < self.max_num_seqs:  # running 队列不为空，且当前运行的序列数小于最大请求数
            seq = self.running.popleft() # 获取运行队列中的第一个序列，只看队头，LIFO 队列，后进先出
            while not self.block_manager.can_append(seq): # 如果不能分配 KV cache，则抢占序列
                if self.running: # 如果运行队列不为空，则抢占序列
                    self.preempt(self.running.pop()) # 抢占运行队列中的最后一个序列
                else:
                    self.preempt(seq) # 抢占当前序列
                    break
            else:
                num_seqs += 1 # 当前运行的序列数加1，num_seqs 表示当前运行的序列数
                self.block_manager.may_append(seq) # 尝试分配 KV cache
                scheduled_seqs.append(seq) # 将序列添加到需要运行的序列列表，假设取出来 ABC
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs)) # 把本轮从 running 队头取出来的序列，再按原顺序放回队头。顺序不变。取出来 ABC，放回去还是 ABC
        return scheduled_seqs, False # 返回需要运行的序列列表和是否是 prefill

    def preempt(self, seq: Sequence):  # 抢占
        '''
        抢占序列，将序列状态设置为等待中，并从运行队列中移除，添加到等待队列中
        Args:
            seq: 序列
        
        Notes:
            1. 状态改回 waiting
            2. 回收这条序列当前占的 block
            3. 放回 waiting 队头
        '''
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        '''
        后处理，更新序列状态，释放 KV cache
        Args:
            seqs: 序列列表
            token_ids: token_ids 列表
        Returns:
            list[bool]: 是否完成列表
        '''
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 把新 token 接到 seq.token_ids 尾部
            # 命中 EOS，且 ignore_eos=False 或者已经生成到 max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
