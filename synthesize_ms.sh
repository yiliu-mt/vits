#!/bin/bash

config=exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022.json
test_file=exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test.txt
model=logs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/G_380000.pth
output_dir=Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

CUDA_VISIBLE_DEVICES=4 python inference_ms.py -c $config -t $test_file -m $model -o $output_dir
