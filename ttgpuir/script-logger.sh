./run.sh loop-pipeline-1.mlir -tritongpu-assign-latencies
./run.sh ./loop-pipeline-1.afterpass.-tritongpu-assign-latencies.mlir -tritongpu-schedule-loops
./run.sh loop-pipeline-1.afterpass.-tritongpu-assign-latencies.afterpass.-tritongpu-schedule-loops.mlir -tritongpu-pipeline=num-stages=3
./run.sh pipeline-schedule-loop-1.mlir -tritongpu-schedule-loops
./run.sh pipeline-schedule-loop-1.afterpass.-tritongpu-schedule-loops.mlir -tritongpu-pipeline=num-stages=2
