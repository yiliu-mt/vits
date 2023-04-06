#!/bin/bash

# Use 4 GPUs to train the model
# The model will be generated in "logs/musa/"
CUDA_VISIBLE_DEVICES=0 python train_ms.py \
    -c exp/2023Q1/musa/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train.json \
    -m musa
