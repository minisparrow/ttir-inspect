
import torch
import triton
import triton.language as tl

# ==========================================
# 1. PyTorch 参考实现 (Specification)
# ==========================================
def add_vec_spec(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [N0], y: [N1]
    # x[None, :] -> [1, N0]
    # y[:, None] -> [N1, 1]
    # 结果 -> [N1, N0]
    return x[None, :] + y[:, None]

# ==========================================
# 2. Triton Kernel 实现
# ==========================================
@triton.jit
def add_vec_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    # -----------------------------------------------------------
    # 为了处理超过一个 Block 的数据，我们需要加上 Program ID (pid)
    # 但在当前题目 N=32 的情况下，如果 BlockSize=32，pid 会始终为 0
    # -----------------------------------------------------------
    # pid_0 = tl.program_id(0) # 对应 x 维度 (列)
    # pid_1 = tl.program_id(1) # 对应 y 维度 (行)

    # 生成当前 Block 的索引范围
    # 注意：加上 pid * B 偏移量是为了支持大矩阵平铺 (Tiling)
    # offset_x = pid_0 * B0 + tl.arange(0, B0)
    # offset_y = pid_1 * B1 + tl.arange(0, B1)
    offset_x = tl.arange(0, B0)
    offset_y = tl.arange(0, B1)

    # 创建 Mask 防止越界 (虽然 N=32, B=32 时不需要，但这是好习惯)
    mask_x = offset_x < N0
    mask_y = offset_y < N1

    # 加载 x 和 y
    # x 对应列 (N0)，y 对应行 (N1)
    x = tl.load(x_ptr + offset_x, mask=mask_x)
    y = tl.load(y_ptr + offset_y, mask=mask_y)

    # 计算广播加法
    # x[None, :] 形状变成 [1, B0]
    # y[:, None] 形状变成 [B1, 1]
    # z 形状变成 [B1, B0]
    z = x[None, :] + y[:, None]

    # 计算输出矩阵 Z 的写入位置
    # Z 是 (N1, N0) 的矩阵，在内存中展平存储
    # 指针位置 = 行索引 * 步长(N0) + 列索引
    # 这里利用广播机制生成二维的 offset 矩阵
    offset = offset_y[:, None] * N0 + offset_x[None, :]

    # 生成二维 mask
    mask = mask_y[:, None] & mask_x[None, :]

    # 存储结果
    tl.store(z_ptr + offset, z, mask=mask)

# ==========================================
# 3. 启动与测试代码
# ==========================================
def test(kernel, spec, nelem):
    print(f"运行设备检测: {'CUDA' if torch.cuda.is_available() else 'CPU (Triton Interpreter)'}")

    # 设定维度
    N0 = nelem["N0"] # x 的长度 (列数)
    N1 = nelem["N1"] # y 的长度 (行数)

    # 设定 Block Size
    # 因为输入是 32，我们设为 32，这样 1 个 Block 就能处理完
    # 如果设为 16，Triton 会启动 2x2=4 个 Block
    B0 = 32
    B1 = 32

    # 准备数据
    torch.manual_seed(0)
    if torch.cuda.is_available():
        x = torch.randn(N0, device='cuda', dtype=torch.float32)
        y = torch.randn(N1, device='cuda', dtype=torch.float32)
    else:
        # 如果没有 GPU，使用模拟器模式
        x = torch.randn(N0, device='cpu', dtype=torch.float32)
        y = torch.randn(N1, device='cpu', dtype=torch.float32)

    # 分配输出空间 (N1 行, N0 列)
    z = torch.empty((N1, N0), device=x.device, dtype=x.dtype)

    # 计算 Grid
    # 这是告诉 GPU 需要启动多少个 Block
    # grid 是一个 tuple，可以是 1D, 2D, 或 3D
    grid = lambda meta: (triton.cdiv(N0, meta['B0']), triton.cdiv(N1, meta['B1']))

    print(f"Input sizes: x={x.shape}, y={y.shape}")
    print(f"Output size: {z.shape}")
    print(f"Block sizes: B0={B0}, B1={B1}")

    # 运行 Kernel
    kernel[grid](x, y, z, N0, N1, B0=B0, B1=B1)

    # 运行 PyTorch 参考实现
    z_ref = spec(x, y)

    # 验证
    if torch.allclose(z, z_ref):
        print("✅ 测试通过! Triton 结果与 PyTorch 一致。")
        print("Sample Output (Top-Left 5x5):\n", z[:5, :5].cpu().numpy())
    else:
        print("❌ 测试失败!")
        diff = (z - z_ref).abs().max()
        print(f"最大误差: {diff}")

if __name__ == "__main__":
    test(add_vec_kernel, add_vec_spec, nelem={"N0": 32, "N1": 32})
