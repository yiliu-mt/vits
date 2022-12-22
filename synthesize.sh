#!/bin/bash

config=configs/baker_inhouse.json
test_file=filelists/baker_valid_inhouse_blank.txt
model=logs/baker_inhouse/G_200000.pth
output_dir=baker_inhouse

python inference.py -c $config -t $test_file -m $model -o $output_dir
