#!/bin/bash

# 检查是否提供了输入文件
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <输入.mlir> <nv/amd>"
    exit 1
fi

INPUT_FILE=$1
backend=$2
# 基于输入文件名创建输出文件名
OUTPUT_FILE="${INPUT_FILE%.mlir}.${backend}.out.mlir"

# 检查 triton-opt 命令是否存在
if ! command -v triton-opt &> /dev/null
then
    echo "错误: 未找到 'triton-opt' 命令。"
    echo "请确保它已安装并在您的 PATH 环境变量中。"
    exit 1
fi

echo "正在处理 $INPUT_FILE..."
# 运行 triton-opt 命令并将输出重定向到文件

if [ "${backend}" == "nv" ]; then
    echo "nvidia backend..."
    #triton-opt --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm "$INPUT_FILE" > "$OUTPUT_FILE"
    triton-opt --triton-nvidia-tma-lowering "$INPUT_FILE" > "$OUTPUT_FILE"
else
    echo "amd backend..."
    triton-opt --convert-triton-amdgpu-to-llvm=arch=gfx942 "$INPUT_FILE" > "$OUTPUT_FILE"
fi

echo "成功生成 $OUTPUT_FILE"
