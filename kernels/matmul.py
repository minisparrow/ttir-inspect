import torch
import triton
import triton.language as tl

def get_autotune_config():
    configs = []
    
    # 1. 定义搜索空间
    block_m_range = [64, 128, 256]
    block_n_range = [64, 128, 256]
    block_k_range = [32, 64]
    num_stages_range = [2, 3, 4, 5]
    num_warps_range = [4, 8]
    
    for m in block_m_range:
        for n in block_n_range:
            for k in block_k_range:
                for stages in num_stages_range:
                    for warps in num_warps_range:
                        
                        # 2. 剪枝策略 (Pruning Rules)
                        # 排除明显不合理的配置，减少编译时间
                        
                        # 规则 A: 块太小不值得用太多 warps
                        if m * n < 256 * 64 and warps == 8:
                            continue
                            
                        # 规则 B: 共享内存限制估算 (非常重要!)
                        # A100 Shared Memory 约为 164KB。
                        # 粗略估算: (M*K + K*N) * sizeof(float16) * stages
                        # 如果超过硬件限制，Triton 编译会报错或跳过，但最好自己先筛掉
                        # float16 = 2 bytes
                        shmem_usage = (m * k + k * n) * 2 * stages
                        if shmem_usage > 160 * 1024: # 留一点余量
                            continue

                        configs.append(triton.Config(
                            {'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n, 'BLOCK_SIZE_K': k, 'GROUP_SIZE_M': 8},
                            num_stages=stages,
                            num_warps=warps
                        ))
    return configs

# 使用方法
# @triton.autotune(
#     configs=get_autotune_config(), # 注入生成的几十种配置
#     key=['M', 'N', 'K']
# )

@triton.autotune(
    configs=[
        # Config 结构: 
        #   meta: 传递给 kernel 的 constexpr 参数 (块大小等)
        #   num_warps: 线程束数量 (一个 warp 是 32 线程)
        #   num_stages: 软件流水线级数 (用于隐藏内存延迟，需 Ampere+ GPU 支持较好)
        
        # # 针对大矩阵的高吞吐配置 (占用更多共享内存)
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # 
        # # 针对中等矩阵的平衡配置
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
        # # 针对较小矩阵或保守配置
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    # key: 当这些参数发生变化时，Triton 会重新评估最佳配置
    # 通常矩阵乘法的性能主要取决于 M, N, K 的大小
    key=['M', 'N', 'K'],
)



@triton.jit
def matmul_kernel(
    # 指针参数
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # Stride (步长)，用于计算内存地址
    stride_am, stride_ak,  # A 的 stride
    stride_bk, stride_bn,  # B 的 stride
    stride_cm, stride_cn,  # C 的 stride
    # 元编程参数 (编译时常量)，用于分块大小
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,  # 用于 L2 Cache 优化的参数
):
    # 1. 获取当前程序的 ID (PID)
    pid = tl.program_id(axis=0)
    
    # -----------------------------------------------------------
    # L2 Cache 优化 (Swizzling)
    # 这一段逻辑将 grid 重新排序，使得内存访问更具局部性，提高 L2 Cache 命中率
    # 如果不理解可以先跳过，假设 grid 是一维线性的即可
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # M 方向有多少个块
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # N 方向有多少个块
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # -----------------------------------------------------------

    # 2. 计算内存偏移指针
    # 我们需要创建指向 A 和 B 第一个块的指针
    
    # offs_am: 生成 0 到 BLOCK_SIZE_M 的序列，表示块内的行偏移
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # offs_bn: 生成 0 到 BLOCK_SIZE_N 的序列，表示块内的列偏移
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # offs_k: 生成 0 到 BLOCK_SIZE_K 的序列，表示 K 维度的偏移
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # 计算 A 指针: 
    # Base + (行索引 * 行Stride) + (列索引 * 列Stride)
    # 这里使用了广播机制 [:, None] 将一维向量变成二维，以便生成矩阵块的指针
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # 计算 B 指针:
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. 主循环：沿着 K 维度迭代
    # 初始化累加器，精度通常设为 float32 以避免溢出
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 和 B 的块
        # mask 是用来处理 K 维度不能被 BLOCK_SIZE_K 整除的边界情况
        # other=0.0 表示越界部分填充 0，不影响加法结果
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 矩阵乘法：计算 block_a * block_b 并累加
        accumulator = tl.dot(a, b, accumulator)
        
        # 更新指针，移动到下一个 K 块
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 4. 存储结果
    # 获取 C 的 dtype (通常是 float16)
    c = accumulator.to(tl.float16)
    
    # 计算 C 的内存地址
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # 创建 mask 防止写入越界
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # 写入内存
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # 检查矩阵形状
    assert a.shape[1] == b.shape[0], "Matrix dimensions mismatch"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape

    # 分配结果张量
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 1. 定义分块配置 (超参数)
    # 这些数值通常需要根据 GPU 型号进行调优 (Auto-Tuning)
    # 这里的配置适合大多数情况
    # BLOCK_SIZE_M = 128
    # BLOCK_SIZE_N = 128
    # BLOCK_SIZE_K = 32
    # GROUP_SIZE_M = 8

    # 2. 计算 Grid 大小 (启动多少个 kernel 实例)
    # grid 是一个函数，接收 META 参数并返回元组 (x, y, z)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 
    )

    # 3. 启动 Kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
        # BLOCK_SIZE_M=BLOCK_SIZE_M,
        # BLOCK_SIZE_N=BLOCK_SIZE_N,
        # BLOCK_SIZE_K=BLOCK_SIZE_K,
        # GROUP_SIZE_M=GROUP_SIZE_M
    )

    return c


# 确保在 GPU 上运行
torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)

# 运行 Triton 实现
triton_output = matmul(a, b)

# 运行 PyTorch 官方实现
torch_output = torch.matmul(a, b)

# 验证正确性
print(f"Triton output shape: {triton_output.shape}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Success! Results match.")
else:
    print("❌ Failure! Results do not match.")
    # 打印最大误差
    print(f"Max diff: {torch.max(torch.abs(triton_output - torch_output))}")


import torch
import triton

# 假设你已经定义了上面的 matmul 函数

def test_performance_single(M, N, K):
    print(f"Running benchmark for size ({M}x{N}x{K})...")
    
    # 1. 准备数据
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    # 2. 定义 lambda 表达式以便 benchmark 调用
    triton_op = lambda: matmul(a, b)
    torch_op = lambda: torch.matmul(a, b)
    
    # 3. 使用 triton.testing.do_bench 进行精确计时
    # do_bench 会自动处理 GPU 预热 (warmup) 和多次测量取平均值
    triton_ms = triton.testing.do_bench(triton_op)
    torch_ms = triton.testing.do_bench(torch_op)
    
    # 4. 计算 TFLOPS
    # 矩阵乘法的 FLOPs = 2 * M * N * K
    total_flops = 2 * M * N * K
    triton_tflops = total_flops * 1e-12 / (triton_ms * 1e-3)
    torch_tflops = total_flops * 1e-12 / (torch_ms * 1e-3)
    
    print(f"----------------------------------------------------------")
    print(f"Size: {M}x{N}x{K}")
    print(f"Triton: {triton_ms:.4f} ms | {triton_tflops:.4f} TFLOPS")
    print(f"PyTorch: {torch_ms:.4f} ms | {torch_tflops:.4f} TFLOPS")
    print(f"Speedup (Triton / PyTorch): {torch_ms / triton_ms:.2f}x")
    print(f"----------------------------------------------------------")

# 使用示例
test_performance_single(4096, 4096, 4096)

print(matmul_kernel.best_config)
