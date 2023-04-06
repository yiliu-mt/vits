#!/bin/bash

# Use 4 V100 GPUs to train the model
# The model will load pretrained model from logs/musa_pretrained and 
# output the finetuned model in "logs/musa/"
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/musa/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train_full.json \
    -m musa \
    -p musa_pretrained
