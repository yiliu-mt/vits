#!/bin/bash

config=configs/baker_wetts.json
test_file=filelists/baker_valid_wetts.txt
model=logs/baker_wetts/G_240000.pth
output_dir=baker_wetts

python inference.py -c $config -t $test_file -m $model -o $output_dir
