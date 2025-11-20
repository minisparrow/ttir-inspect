
import torch
import triton
import triton.language as tl

# 1. 定义 PyTorch 的参考实现 (用于验证结果)
def add2_spec(x: torch.Tensor) -> torch.Tensor:
    return x + 10.0

# 2. Triton Kernel 实现 (补全了你的代码)
@triton.jit
def add_mask2_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    # 获取当前程序的 block ID
    pid = tl.program_id(0)

    # 生成当前 block 处理的数据范围 (0 到 B0-1)
    # 注意：range 是 python 关键字，建议改名为 range_vals 或 offs
    range_vals = tl.arange(0, B0)

    # 计算全局偏移量
    offset = pid * B0 + range_vals

    # 创建掩码：防止处理超过 N0 长度的数据
    # 例如 N0=200, B0=256, 那么索引 200-255 的 mask 为 False
    mask = offset < N0

    # 加载数据，对于越界部分，mask 会阻止非法内存访问 (默认填充0，但这里不影响)
    x = tl.load(x_ptr + offset, mask=mask)

    # 计算
    z = x + 10.0

    # 写回数据，同样需要 mask 保护
    tl.store(z_ptr + offset, z, mask=mask)

# 3. 宿主函数 (Launcher)：计算 Grid 并启动 Kernel
def add2_triton(x: torch.Tensor):
    N0 = x.numel()
    z = torch.empty_like(x) # 分配输出显存

    # 设定 Block Size (通常设为 2 的幂次，例如 128, 256, 1024)
    # 这里为了演示 mask 的效果，我们设为 256，大于输入的 200
    BLOCK_SIZE = 256

    # 计算 Grid 大小：(N0 + BLOCK_SIZE - 1) // BLOCK_SIZE
    # triton.cdiv 是向上取整除法
    grid = lambda meta: (triton.cdiv(N0, meta['B0']),)

    # 启动 Kernel
    add_mask2_kernel[grid](x, z, N0, B0=BLOCK_SIZE)

    return z

# 4. 测试与验证
def test():
    # 检查是否有 GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到 CUDA 设备 (GPU)")
        return

    torch.manual_seed(0)

    # 设置数据大小 N0 = 200，正如你代码中要求的
    N0 = 200

    # 创建 GPU 上的 Tensor
    x = torch.randn(N0, device='cuda', dtype=torch.float32)

    print(f"Input (前5个): {x[:5].cpu().numpy()}")

    # 运行 PyTorch 参考版
    output_torch = add2_spec(x)

    # 运行 Triton 版
    output_triton = add2_triton(x)

    print(f"Triton Output (前5个): {output_triton[:5].cpu().numpy()}")

    # 验证结果是否一致
    if torch.allclose(output_torch, output_triton):
        print("✅ 测试通过！Triton 结果与 PyTorch 结果一致。")
    else:
        print("❌ 测试失败！结果不一致。")
        diff = (output_torch - output_triton).abs().max()
        print(f"最大误差: {diff}")

if __name__ == "__main__":
    test()
