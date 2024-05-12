#!/bin/bash

# GPU ID
GPU_ID=0

# 获取GPU ID为0的所有进程PID
PIDS=$(nvidia-smi | grep " $GPU_ID " | awk '{print $5}')

for PID in $PIDS; do
    if [[ $PID =~ ^[0-9]+$ ]]; then  # 确保PID是数字
        echo "Killing PID $PID"
        kill -9 $PID
    else
        echo "Skipping non-PID value: $PID"
    fi
done

