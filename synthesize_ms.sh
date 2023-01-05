#!/bin/bash

config=exp/aishell3/1226/configs/aishell3_trim.json
test_file=exp/aishell3/1226/preprocessed_data/val.txt
model=logs/aishell3_trim/G_200000.pth
output_dir=aishell3_trim

python inference_ms.py -c $config -t $test_file -m $model -o $output_dir
