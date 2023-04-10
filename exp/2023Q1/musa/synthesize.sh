#!/bin/bash

config=/nfs1/yi.liu/tts/vits/exp/2023Q1/jams_v0/0308/tuning_jams_rx_full/config.json
model=/nfs1/yi.liu/tts/vits/exp/2023Q1/jams_v0/0308/tuning_jams_rx_full/G_300000.pth
lexicon=lexicon/v1.9/bilingual_serving_dict_er3.txt
speaker_id=222

test_data=testdata/jams/jams_test_ssmlv2.txt
output_dir=audio_outs
mkdir -p $output_dir

export PYTHONPATH=.

# The audio will be generated in "audio_outs"
CUDA_VISIBLE_DEVICES=0 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir
