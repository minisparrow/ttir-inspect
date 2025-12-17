
import torch
import triton
import triton.language as tl

@triton.jit
def figure4_shuffle_kernel(input_ptr, output_ptr):
    # --- 1. 构造输入布局 (4x2) ---
    # 行索引: 0..3 (对应 Thread)
    # 列索引: 0..1 (对应 Reg)
    row_idx = tl.arange(0, 4)
    col_idx = tl.arange(0, 2)

    # 构造 Layout A (4x2) 的偏移量
    # Input memory layout: 连续存储 [0, 1, 2, 3, 4, 5, 6, 7]
    # stride 为 2
    offs_a = row_idx[:, None] * 2 + col_idx[None, :]

    # Load: x shape = (4, 2)
    x = tl.load(input_ptr + offs_a)

    # --- 2. Transpose (触发 Shuffle) ---
    # y shape = (2, 4)
    y = tl.trans(x)

    # --- 3. 构造输出布局 (2x4) ---
    # 为了存回 y，我们需要一个 (2, 4) 的指针块
    # 新的行（原列）: 0..1
    # 新的列（原行）: 0..3

    # 定义输出内存的 stride，假设我们要把转置结果连续写入内存
    # 输出形状是 2x4，所以 stride 是 4
    out_row_idx = tl.arange(0, 2)
    out_col_idx = tl.arange(0, 4)

    # 构造 (2, 4) 的偏移量
    offs_out = out_row_idx[:, None] * 4 + out_col_idx[None, :]

    # Store: 现在指针 shape (2, 4) 和数据 y shape (2, 4) 匹配了
    tl.store(output_ptr + offs_out, y)

# 验证代码
if __name__ == "__main__":
    src = torch.arange(8, dtype=torch.float32, device='cuda')
    # 目标内存大小一样，也是8个元素
    dst = torch.zeros(8, dtype=torch.float32, device='cuda')

    # 启动 Kernel，Grid 设为 (1,) 保证在一个 Warp 内
    figure4_shuffle_kernel[(1, 1, 1)](src, dst, num_warps=1)

    print("Input: ", src.cpu().numpy())
    print("Output:", dst.cpu().numpy())

    # 验证逻辑：
    # Input (4x2):
    # [[0, 1],
    #  [2, 3],
    #  [4, 5],
    #  [6, 7]]
    #
    # Transpose (2x4):
    # [[0, 2, 4, 6],
    #  [1, 3, 5, 7]]
    #
    # Output memory (linearized): [0, 2, 4, 6, 1, 3, 5, 7]

