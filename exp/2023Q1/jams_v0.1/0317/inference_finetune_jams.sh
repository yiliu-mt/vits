# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

config=/home/ubuntu/vits_jams/logs/exp/2023Q1/jams_v0.1/0317/tuning_jams_rx_full_sdp/config.json
model=/home/ubuntu/vits_jams/logs/exp/2023Q1/jams_v0.1/0317/tuning_jams_rx_full_sdp/G_220000.pth
lexicon=lexicon/v1.9/bilingual_serving_dict_er3.txt
speaker_id=222

test_data=testdata/jams/test.txt
# test_data=testdata/jams/jams_test_ssmlv2.txt
output_dir=/home/ubuntu/vits_jams/logs/exp/2023Q1/jams_v0.1/0317/tuning_jams_rx_full_sdp/test

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=6 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir
