
import torch
import triton
import triton.language as tl

# 设置环境变量以查看 PTX (如果环境支持)
# import os
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def warp_reduction_kernel(input_ptr, output_ptr):
    # 1. 每个线程加载一个数据
    # 我们用 range(32) 对应一个 Warp 的 32 个线程
    offs = tl.arange(0, 32)
    x = tl.load(input_ptr + offs)

    # 2. 执行归约求和
    # 这在底层就是 Figure 4 所示的数学过程：
    # 每一轮，线程 i 和线程 i ^ mask 交换数据并相加
    # 这里的 mask 就是数学推导中的 span(G) 的基向量
    y = tl.sum(x, axis=0)

    # 3. 只有线程 0 输出结果
    if tl.program_id(0) == 0:
        tl.store(output_ptr, y)

def check_ptx():
    src = torch.arange(32, dtype=torch.float32, device='cuda')
    dst = torch.zeros(1, dtype=torch.float32, device='cuda')

    # 编译并获取 PTX 代码
    kernel = warp_reduction_kernel[(1,)](src, dst, num_warps=1)

    # 这是一个 Hack 方法，用于在 Python 中直接打印 PTX
    # 注意：这依赖于 Triton 的内部结构，不同版本可能不同
    # 通常建议在命令行使用 TRITON_DUMP_DIR 查看
    try:
        # 尝试访问缓存 (适用于部分版本)
        for key, val in kernel.cache.items():
            print(val.asm['ptx'])
            break
    except:
        print("无法直接打印 PTX，请使用环境变量 TRITON_DUMP_DIR=./dump 运行脚本，并在 dump 文件夹中查看 .ptx 文件")

if __name__ == "__main__":
    check_ptx()
