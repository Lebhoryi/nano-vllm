# Torch multi-threading APIs
_Exported on 3/11/2026 at 16:27:20 GMT+8 from Cursor (2.5.26)_

---

**User**

涉及到的 torch 多线程相关的api有哪些

---

**Cursor**

我先在仓库里全局检索一遍和 PyTorch 线程/并行有关的调用，再给你按“项目里已用到”和“常用可用 API”两部分整理。

在你这个项目里，**严格说主要是多进程/分布式并行，不是 `torch` CPU 多线程**。我先按“项目已用到”给你列，再补“真正多线程控制 API”。

- **项目里已用到（并行相关）**
  - `torch.multiprocessing.get_context("spawn")`（`nanovllm/engine/llm_engine.py`）
  - `ctx.Process(...)` / `process.start()` / `process.join()`（同上）
  - `ctx.Event()` + `event.wait()/set()/clear()`（跨进程同步，`llm_engine.py` + `model_runner.py`）
  - `torch.multiprocessing.spawn(...)`（`test.py`）
  - `torch.distributed.init_process_group(...)` / `destroy_process_group(...)`（`model_runner.py`、`test.py`）
  - `torch.distributed.barrier()` / `all_reduce()` / `gather()` / `get_rank()` / `get_world_size()`（`model_runner.py`、`nanovllm/layers/*.py`、`nanovllm/models/qwen3.py`）

- **PyTorch 真正“多线程”常用 API（你项目里目前没显式用）**
  - `torch.set_num_threads(n)` / `torch.get_num_threads()`：控制/查询 **intra-op**（算子内部）CPU 线程数
  - `torch.set_num_interop_threads(n)` / `torch.get_num_interop_threads()`：控制/查询 **inter-op**（算子之间）线程并行
  - 相关环境变量：`OMP_NUM_THREADS`、`MKL_NUM_THREADS`（常用于限制底层线程库线程数）

如果你愿意，我可以再给你一份“**nano-vllm 场景下推荐线程/进程参数**”（比如单机多卡推理时该怎么配 `tensor_parallel_size` + OMP/MKL 线程数）。

