#!/bin/bash

python make_shard_data.py --config configs/baker_base.json --num_threads 16 \
    filelists/baker_train.txt data/train data/train.txt

python make_shard_data.py --config configs/baker_base.json --num_threads 16 \
    filelists/baker_valid.txt data/valid data/valid.txt
