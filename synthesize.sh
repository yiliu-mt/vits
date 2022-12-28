#!/bin/bash

config=exp/baker/1221/configs/baker_trim.json
test_file=exp/baker/1221/preprocessed_data/val.txt
model=logs/baker_trim/G_200000.pth
output_dir=baker_trim

python inference.py -c $config -t $test_file -m $model -o $output_dir
