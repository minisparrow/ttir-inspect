#!/bin/bash

# 检查是否提供了输入文件
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <输入.mlir> <passname>"
    exit 1
fi

INPUT_FILE=$1
passname=$2
# 基于输入文件名创建输出文件名
OUTPUT_FILE="${INPUT_FILE%.mlir}.afterpass.${passname}.mlir"

# 检查 triton-opt 命令是否存在
if ! command -v triton-opt &> /dev/null
then
    echo "错误: 未找到 'triton-opt' 命令。"
    echo "请确保它已安装并在您的 PATH 环境变量中。"
    exit 1
fi

echo "正在处理 $INPUT_FILE..."
# 运行 triton-opt 命令并将输出重定向到文件

echo "compiling pass..."
triton-opt "$passname" "$INPUT_FILE" > "$OUTPUT_FILE"

echo "成功生成 $OUTPUT_FILE"
