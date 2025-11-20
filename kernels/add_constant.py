
import torch
import triton
import triton.language as tl

# Check if GPU is available
assert torch.cuda.is_available(), "你需要一个 GPU 来运行 Triton 代码"

# --------------------------------------------------------
# 1. 定义 Spec (参考实现)
# --------------------------------------------------------
def add_spec(x: torch.Tensor) -> torch.Tensor:
    """
    这是 PyTorch 的参考实现（即 Spec）。
    它定义了我们希望 Triton Kernel 做什么：给 x 加上 10。
    """
    return x + 10.0

# --------------------------------------------------------
# 2. 定义 Triton Kernel
# --------------------------------------------------------
@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    """
    Triton Kernel 实现。
    x_ptr: 输入张量的指针
    z_ptr: 输出张量的指针
    N0: 元素总数 (在这个特定的 Puzzle 中是 32)
    B0: Block Size (块大小，在这个 Puzzle 中通常等于 N0)
    """
    # 生成索引范围 [0, 1, ..., B0-1]
    # 注意：range 是 Python 关键字，虽然在这里可以用，但通常建议用 offsets
    pid = tl.program_id(0) # 获取当前程序的 ID

    # 计算当前 block 处理的数据范围
    # 这里的逻辑是针对 Puzzle 特化的 (N0=32, B0=32)，
    # 如果处理更大的数据，通常是 pid * B0 + tl.arange(0, B0)
    range_vals = tl.arange(0, B0)

    # 加上 pid 偏移量 (为了通用性，虽然这里只有 1 个 block)
    offsets = pid * B0 + range_vals

    # 掩码 (Mask)：防止访问越界。
    # 虽然题目中 N0=32 刚好填满，但在 Triton 中加上 mask 是好习惯
    mask = offsets < N0

    # 从显存加载数据
    x = tl.load(x_ptr + offsets, mask=mask)

    # --- Finish me! (你填写的逻辑) ---
    z = x + 10.0
    # -------------------------------

    # 将结果写回显存
    tl.store(z_ptr + offsets, z, mask=mask)

# --------------------------------------------------------
# 3. 驱动代码 (模拟 test 函数)
# --------------------------------------------------------
def test_implementation():
    # 设置参数
    N0 = 32
    BLOCK_SIZE = 32  # B0

    # 准备数据 (在 GPU 上)
    torch.manual_seed(0)
    x = torch.randn(N0, device='cuda', dtype=torch.float32)
    z_ref = torch.empty_like(x) # 用于存放 spec 的结果
    z_tri = torch.empty_like(x) # 用于存放 kernel 的结果

    # 1. 运行 Spec (参考答案)
    z_ref = add_spec(x)

    # 2. 运行 Triton Kernel
    # Grid 定义了我们要启动多少个处理单元。
    # ceil(N0 / BLOCK_SIZE)
    grid = lambda meta: (triton.cdiv(N0, meta['B0']), )

    # 启动 Kernel
    add_kernel[grid](x, z_tri, N0, B0=BLOCK_SIZE)

    # 3. 验证结果
    print(f"Input (前5个): {x[:5].cpu().numpy()}")
    print(f"Spec Output (前5个): {z_ref[:5].cpu().numpy()}")
    print(f"Triton Output (前5个): {z_tri[:5].cpu().numpy()}")

    # 使用 allclose 比较浮点数
    if torch.allclose(z_ref, z_tri):
        print("\n✅ 成功! Triton Kernel 的结果与 Spec 一致。")
    else:
        print("\n❌ 失败! Triton Kernel 的结果不正确。")
        print(f"最大误差: {(z_ref - z_tri).abs().max().item()}")

if __name__ == "__main__":
    test_implementation()
