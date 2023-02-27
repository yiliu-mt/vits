#!/bin/bash

config=exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10.json
test_file=exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_musha.txt
model=logs/exp/2023Q1/0112/tuning_musha_mt10/G_80000.pth
output_dir=exp/2023Q1/0112/audio_out/tuning_musha_mt10
CUDA_VISIBLE_DEVICES=7 python inference_ms.py -c $config -t $test_file -m $model -o $output_dir

config=exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json
test_file=exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_mita.txt
model=logs/exp/2023Q1/0112/tuning_mita_bc_chatbot10/G_80000.pth
output_dir=exp/2023Q1/0112/audio_out/tuning_mita_bc_chatbot10
CUDA_VISIBLE_DEVICES=7 python inference_ms.py -c $config -t $test_file -m $model -o $output_dir
