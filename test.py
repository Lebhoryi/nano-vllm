import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 初始化分布式环境
def init_dist(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size,
    )

# 每个进程的核心逻辑
def worker(rank, world_size):
    # 1. 初始化分布式环境
    print(f"rank:{rank} world_size:{world_size}")
    init_dist(rank, world_size)
    
    # 2. 模拟每个进程的计算结果（张量形状必须一致）
    # rank0: [0,0], rank1: [1,1], rank2: [2,2], rank3: [3,3]
    local_tensor = torch.ones(2, 2).to(rank) * rank
    print(f"[Rank {rank}] 本地张量: {local_tensor.cpu().numpy()}")
    
    # 3. 准备收集列表（仅目标进程dst=0需要初始化）
    gather_list = None
    if rank == 0:
        # 初始化gather_list：长度=world_size，每个元素是和local_tensor同形状的空张量
        gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    
    # 4. 执行gather：将所有进程的local_tensor收集到rank0的gather_list中
    dist.gather(
        tensor=local_tensor,
        gather_list=gather_list,
        dst=0  # 收集到主进程（rank0）
    )
    
    # 5. 目标进程打印收集结果
    if rank == 0:
        print("\n[Rank 0] 收集到的所有张量：")
        for i, tensor in enumerate(gather_list):
            print(f"  来自Rank {i}: {tensor.cpu().numpy()}")
    
    # 6. 清理
    dist.destroy_process_group()

# 主函数启动多进程
if __name__ == "__main__":
    world_size = 2  # 4个进程（4张GPU）
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)