
import torch
import triton
import triton.language as tl

def mul_relu_block_spec(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x[None, :] * y[:, None])


@triton.jit

def mul_relu_block_kernel(x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
 pid_0 = tl.program_id(0)
 pid_1 = tl.program_id(1)
 offset_x = pid_0 * B0 + tl.arange(0, B0)
 offset_y = pid_1 * B1 + tl.arange(0, B1)
 offset_z = offset_y[:, None] * N0 + offset_x[None, :]
 mask_x = offset_x < N0
 mask_y = offset_y < N1
 mask = mask_x[None, :] & mask_y[:, None]
 x = tl.load(x_ptr + offset_x, mask=mask_x, other=0.0)
 y = tl.load(y_ptr + offset_y, mask=mask_y, other=0.0)
 z = x[None, :] * y[:, None]
 z = tl.maximum(z, 0) # pay attention to here: not tl.max but tl.maximum
 tl.store(z_ptr + offset_z, z, mask=mask)
 return



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
    mul_relu_block_kernel[grid](x, y, z, N0, N1, B0=B0, B1=B1)

    # 运行 PyTorch 参考实现
    z_ref = mul_relu_block_spec(x, y)

    # 验证
    if torch.allclose(z, z_ref):
        print("✅ 测试通过! Triton 结果与 PyTorch 一致。")
        print("Sample Output (Top-Left 5x5):\n", z[:5, :5].cpu().numpy())
    else:
        print("❌ 测试失败!")
        diff = (z - z_ref).abs().max()
        print(f"最大误差: {diff}")

if __name__ == "__main__":
    test(mul_relu_block_kernel, mul_relu_block_spec, nelem={"N0": 100, "N1": 90})
