from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    '''
    KV cache block，仅保存元数据，真正的 kv 数据在 ModelRunner.allocate_kv_cache() 中分配。
    一个 Block 表示一个物理块，物理块是连续的，但逻辑块是分开的。
    Args:
        block_id: block id
    '''
    def __init__(self, block_id):
        self.block_id = block_id  # 物理块 id
        self.ref_count = 0  # 引用计数，有多少条序列正在引用这个块
        self.hash = -1  # 前缀哈希值，用于判断两个逻辑块是否相同
        self.token_ids = []  # 物理块中的 token ids

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1 # 引用计数初始化为 1，因为自己本身就引用自己
        self.hash = -1 # 前缀哈希值初始化为 -1
        self.token_ids = [] # 物理块中的 token ids 清空


class BlockManager:
    '''
    KV cache 块管理器，负责分配和回收 KV cache block
    Args:
        num_blocks: 总 block 数
        block_size: block 大小
    
    Notes:
        不保存真正的 KV 张量，只管理“KV block 的元数据和分配关系”。仅维护元数据，真正的 kv 数据在 ModelRunner.allocate_kv_cache() 中分配。
        1. 哪些 block 空闲、哪些在使用
        2. 每条 Sequence 的 block_table
        3. 哪些前缀 block 可以复用
        4. block 的引用计数 ref_count
    '''
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size # block 大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # 所有 block 对象列表
        self.hash_to_block_id: dict[int, int] = dict() # 前缀哈希值到 block id 的映射，dict 字典，key 是前缀哈希值，value 是 block id
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲 block id 队列，FIFO 队列，先进先出
        self.used_block_ids: set[int] = set() # 使用中 block id 集合，set 集合，无序不重复

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        '''到当前块为止的完整前缀哈希：当前块内容 + 前一个块的哈希值'''
        # 1. 创建XXH64哈希对象，XXH64是高性能非加密哈希算法，速度远快于MD5/SHA
        h = xxhash.xxh64() 
        
        # 2. 如果传入了有效的前缀哈希（非-1），则将前缀哈希转为小端字节并更新到哈希对象
        if prefix != -1: 
            # prefix.to_bytes(8, "little")：将整数前缀哈希转为8字节的小端序字节串
            # 8字节对应XXH64的输出长度（64位），小端序是常见的存储/传输格式
            h.update(prefix.to_bytes(8, "little")) 
        
        # 3. 将当前块的token_ids转为字节串，更新到哈希对象
        # np.array(token_ids).tobytes()：将int列表转为连续的字节数组（比Python原生list转字节更高效）
        h.update(np.array(token_ids).tobytes()) 
        
        # 4. 返回哈希值的整数形式（而非字节/十六进制），便于后续计算和存储
        return h.intdigest() 

    def _allocate_block(self, block_id: int) -> Block:
        '''
        分配一个 block, 从 free_block_ids 队列中取出一个空闲 block，并重置 block
        Args:
            block_id: block id
        Returns:
            Block: block 对象
        '''
        block = self.blocks[block_id] # 获取 block 对象
        assert block.ref_count == 0  # 引用计数必须为 0，没有被引用，通常用于初始化
        block.reset() # 重置 block，ref_count 设置 1，并清除旧元数据
        self.free_block_ids.remove(block_id) # 从空闲 block id 队列中移除
        self.used_block_ids.add(block_id) # 添加到使用中 block id 集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        '''
        回收一个 block, 从使用中 block id 集合中移除，添加到空闲 block id 队列
        Args:
            block_id: block id
        Returns:
            Block: block 对象
        Notes:
            不会清空 hash_to_block_id，也不会主动删除旧哈希键
        '''
        assert self.blocks[block_id].ref_count == 0 # 引用计数必须为 0，没有被引用，就失去了价值
        self.used_block_ids.remove(block_id) # 从使用中 block id 集合中移除
        self.free_block_ids.append(block_id) # 添加到空闲 block id 队列

    def can_allocate(self, seq: Sequence) -> bool:
        '''检查是否可以分配一个 block'''
        return len(self.free_block_ids) >= seq.num_blocks # 空闲 block 数 >= 序列需要的 block 数

    def allocate(self, seq: Sequence):
        '''
        给一条新序列建立 block_table，并尽可能复用已有前缀块。
        1. 遍历序列的每个 block，计算前缀哈希值
        2. 检查哈希值是否在 hash_to_block_id 中，如果存在，则使用已有的 block
        3. 如果不存在，则从 free_block_ids 队列中取出一个空闲 block，并重置 block
        4. 更新 block 的元数据
        5. 更新 hash_to_block_id 的映射
        6. 更新序列的 block_table
        Args:
            seq: 序列
        Returns:
            Block: block 对象
        '''
        assert not seq.block_table # 序列没有 block_table，说明是新序列，第一次分配 KV cache
        h = -1 # 前缀哈希值，初始化
        cache_miss = False # 是否缓存未命中，初始化
        for i in range(seq.num_blocks):  # 遍历序列中的逻辑块
            token_ids = seq.block(i)
            # 只对完整 block 计算哈希值，如果是最后一个 block，则不计算哈希值
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) # 获取 block id，如果哈希值不存在，则返回 -1
            # 找不到对应 block，或者 token ids 不相同，则缓存未命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中，为了找到连续公共前缀缓存
            if cache_miss: # 缓存未命中，则分配一个新 block
                block_id = self.free_block_ids[0] # 从空闲 block id 队列中取出一个空闲 block
                block = self._allocate_block(block_id) # 分配一个新 block
            else:
                seq.num_cached_tokens += self.block_size
                # 如果正在使用，则计数加一，否则分配一个新的 block，激活它
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1: # 如果哈希值不为 -1，则更新 block 的元数据
                block.update(h, token_ids) # 更新 block 的元数据
                self.hash_to_block_id[h] = block_id # 更新 hash_to_block_id 的映射
            seq.block_table.append(block_id) # 更新序列的 block_table

    def deallocate(self, seq: Sequence):
        '''
        回收一条序列占用的所有 block
        Args:
            seq: 序列
        '''
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1 # 引用计数减一
            if block.ref_count == 0: # 引用计数为 0，则回收 block
                self._deallocate_block(block_id) # 回收 block
        seq.num_cached_tokens = 0 # 已缓存 token 数清零
        seq.block_table.clear() # 清空 block_table

    def can_append(self, seq: Sequence) -> bool:
        '''检查是否可以追加一个 block，如果当前序列长度满足“新 token 会落在新块的第一个位置”'''
        # 判断是否需要新 block，是否是 block 的首个 token id，然后判断是否存在空闲 block_ids
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) # 空闲 block 数 >= 1，则可以追加一个 block

    def may_append(self, seq: Sequence):
        '''尝试追加一个 block'''
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:  # block 里面的首 token id 是新 token，给新 block 占位
            assert last_block.hash != -1 # 最后一个 block 的哈希值不为 -1，说明是新 block
            block_id = self.free_block_ids[0] # 从空闲 block id 队列中取出一个空闲 block
            self._allocate_block(block_id) # 分配一个新 block
            block_table.append(block_id) # 更新序列的 block_table，追加一个新 block
        elif len(seq) % self.block_size == 0: # 最后一个 block 是完整的 block，刚刚被填满，转成可缓存 block
            assert last_block.hash == -1 # 最后一个 block 的哈希值为 -1，说明是新 block
            token_ids = seq.block(seq.num_blocks-1) # 获取最后一个 block 的 token ids
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # 获取倒数第二个 block 的哈希值
            h = self.compute_hash(token_ids, prefix) # 计算最后一个 block 的哈希值
            last_block.update(h, token_ids) # 更新最后一个 block 的元数据
            self.hash_to_block_id[h] = last_block.block_id # 更新 hash_to_block_id 的映射，新的缓存 block，满了，不能再添加
        else:  # 未完成 block
            assert last_block.hash == -1  # 当前最后一个块还是一个普通的 block，没有被复用，还只是个未完成块，说明新 token 落在了旧块的中间
