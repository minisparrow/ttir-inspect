# 或者运行时添加路径
export PYTHONPATH="${PYTHONPATH}:/home/mlrobot/projs/triton-related/triton/python"

export debug=false
export TRITON_BUILD_WITH_O1=$debug
#export DEBUG=$debug
export TRITON_ALWAYS_COMPILE=0
export TRITON_KERNEL_DUMP=$debug
export MLIR_ENABLE_DUMP=$debug
# export TRITON_DUMP_DIR=/home/mlrobot/projs/triton-related/triton/.cache
export TRITON_KERNEL_OVERRIDE=1
export TRITON_FRONT_END_DEBUGGING=1
#export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
