import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """创建 ModelRunner 实例，初始化模型，预热模型，分配 kv cache，捕获 cudagraph，创建共享内存，进入循环，等待 rank0 调用 call(run) 唤醒，继续执行。直到 call(exit) 退出
        Args:
            config: 配置
            rank: 当前进程的 rank，rank 0 是主线程，rank 1 是其他进程
            event: 同步事件，用于同步 rank 0 和 rank 1 的执行
        """
        self.config = config
        hf_config = config.hf_config # 模型配置
        self.block_size = config.kvcache_block_size  # 单个 kv block 的大小
        self.enforce_eager = config.enforce_eager # 是否启用 eager 模式
        self.world_size = config.tensor_parallel_size # 模型并行数
        self.rank = rank # 当前进程的 rank
        self.event = event # 同步事件
        print(f"rank: {rank}")
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 设置当前进程的 GPU 设备，rank 0 -> cuda:0, rank 1 -> cuda:1
        default_dtype = torch.get_default_dtype() # 获取默认的 dtype
        torch.set_default_dtype(hf_config.torch_dtype) # 设置默认的 dtype
        torch.set_default_device("cuda") # 设置默认的设备
        self.model = Qwen3ForCausalLM(hf_config) # 创建模型结构，固定写死 qwen3 模型，注意此时不加载权重
        load_model(self.model, config.model) # 加载模型权重
        self.sampler = Sampler() # 创建采样器
        self.warmup_model() # 预热模型，计算峰值 kv cache 大小。先计算非0的所有rank，然后计算 rank0
        self.allocate_kv_cache() # 分配 kv cache
        if not self.enforce_eager: # 如果不启用 eager 模式
            self.capture_cudagraph() # 捕获 cudagraph
        torch.set_default_device("cpu") # 设置默认的设备，为什么配置默认 cpu？因为模型权重加载后，需要将模型权重从 GPU 移动到 CPU，避免显存占用过高
        torch.set_default_dtype(default_dtype) # 设置默认的 dtype
        if self.world_size > 1: # 如果模型并行数大于 1
            if rank == 0:  # 主线程负责创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # 创建共享内存
                dist.barrier()  # 同步
            else:  # 其他进程负责等待共享内存创建完成
                dist.barrier()  # 等待 rank0 创建共享内存完成
                self.shm = SharedMemory(name="nanovllm") # 连接 rank 0 创建的共享内存
                self.loop()  # rank1 初始化完成后进入 worker 循环，等待 rank0 下发 run/exit 指令；收到 exit 后退出

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0  # 只允许非 0 进程读取共享内存，rank 0 发命令，其他 rank 负责寿命令
        self.event.wait()  # 阻塞等待，直到 rank 0 调用了 event.set()，有新消息了通知，不是执行完成通知
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 共享内存的前 4 个字节专门存消息长度。小端序，int 占 4 个字节，[4: n+4] 真正的消息内容
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 真正内容反序列化，
        self.event.clear()  # 已经读到了，不是执行完了
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0  # 只有 rank 0 可以写共享内存，rank 0 写，其他 rank 读
        data = pickle.dumps([method_name, *args])  # 序列化
        n = len(data)  # 消息长度
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 消息长度写到共享内存的前 4 个字节。小端序，int 占 4 个字节，[4: n+4] 真正的消息内容
        self.shm.buf[4:n+4] = data  # 真正的消息内容序列化
        for event in self.event:
            event.set()  # 广播唤醒所有 worker

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:  # 只有 rank 0 可以写共享内存
            self.write_shm(method_name, *args)  # 写共享内存
        method = getattr(self, method_name, None)  # 获取方法, run / exit，run 和 exit 都是 ModelRunner 的实例方法
        return method(*args)  # run(seqs, is_prefill) / exit()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        # 按当前显存余量和 warmup 峰值，动态算出能放多少 KV block，然后把它分配并接到每一层 Attention 上
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 每个 rank / 每张卡存在多少个 KV heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)  # 每个 head 的维度
        # 2(K+V) * 层数 * block_size * 每卡kv头数 * head_dim * dtype字节数，所以 kv block 在单卡上的内存为 256 * 512 * 2 * 28 = 14680064 个字节
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize  # 单个 kv block 在单卡上的内存
        # 可用预算 = total * gpu_memory_utilization - used - (peak - current)可用预算 = total * gpu_memory_utilization - used - (peak - current)
        # 所以总的 kv block 数 = 可用预算 // 单个 kv block 的内存
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0  # 确保可以分配至少 1 个 kv block
        # 普通场景：[2, 层数, 总的kv block 数 * block_size, head_nums, head_dim]
        # pageattention 的形状为 [2, 层数, 总的kv block 数, block_size, head_nums, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        '''
        构造 block_tables 张量，shape [batch_size, max_num_blocks]，每个元素是序列的 block_table 列表，-1 表示没有 block_table
        Args:
            seqs: 序列列表
        Returns:
            block_tables: 块表张量

        每条序列的 block_table 长度可能不同，因为序列长度不同。把不等长块表补齐成矩阵，然后返回。
        
        Input	                                Output
        seqs[0].block_table = [0, 5, 2]	        [[0, 5, 2],
        seqs[1].block_table = [3, 7]	        [3, 7, -1],
        seqs[2].block_table = [1, 4, 8, 9]	    [1, 4, 8, 9]]
        '''
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 关键逻辑：只处理未缓存部分
        input_ids = [] # 扁平化后的输入 ID 列表
        positions = [] # 扁平化后的位置列表
        cu_seqlens_q = [0] # 每条序列在扁平张量中的前缀和边界（给 varlen attention 用）
        cu_seqlens_k = [0] # 每条序列在扁平张量中的前缀和边界（给 varlen attention 用）
        max_seqlen_q = 0 # batch 内最大查询序列长度
        max_seqlen_k = 0 # batch 内最大键序列长度
        slot_mapping = [] # 新 token 要写入 KV cache 的物理槽位
        block_tables = None # 只有 prefix cache 场景才需要
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])  # [4096,] 等价于 seq.token_ids[seq.num_cached_tokens:]，需要删除缓存的 token ids 长度
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))  # [4096,]
            seqlen_q = seqlen - seq.num_cached_tokens  # 刨除已经缓存过的 token ids，Q，假如 Q token ids 5，已经缓存 3，那么这里就是 2
            seqlen_k = seqlen  # 计算所有长度，此处是 4096
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 更新每条序列在扁平张量中的前缀和边界，注意删除缓存 token ids 长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 更新每条序列在扁平张量中的前缀和边界
            max_seqlen_q = max(seqlen_q, max_seqlen_q) # 更新 batch 内最大查询序列长度，注意删除缓存 token ids 长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k) # 更新 batch 内最大键序列长度
            if not seq.block_table:    # warmup
                continue
            # 为什么 warmup 时 slot_mapping 为空
            # 因为 warmup 序列还没有 block_table，这里只是压一次前向显存，不写 KV。
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) # shape [max_num_batched_tokens] [16384]，注意是 extend，不是 append
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) # shape [max_num_batched_tokens] [16384]，注意是 extend，不是 append
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # shape [num_seqs + 1] [0, 4096, 8192, 12288, 16384]
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # shape [num_seqs + 1] [0, 4096, 8192, 12288, 16384]
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # []
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # 构造 decode 的一步输入
        input_ids = []  # 每条序列的一个 last_token
        positions = []  # 每条序列的一个 last_token 的位置
        slot_mapping = []  # 新 token 要写入 KV cache 的物理槽位
        context_lens = []  # 每条序列的长度
        for seq in seqs:
            input_ids.append(seq.last_token)  # 每条序列的一个 last_token
            positions.append(len(seq) - 1)  # 每条序列的一个 last_token 的位置
            context_lens.append(len(seq))  # 每条序列的长度
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)  # 每条序列的一个 last_token 要写入 KV cache 的物理槽位
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        '''
        准备采样温度，shape [batch_size]，每个元素是序列的采样温度
        Args:
            seqs: 序列列表
        Returns:
            temperatures: 采样温度张量
        
        之所以 batch 内每条都保留一份，是因为不同请求允许不同采样温度。
        '''
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
