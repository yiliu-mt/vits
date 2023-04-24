# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

config=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0419/tuning_jams_mos5_full_sdp/config.json
model=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0418/tuning_jams_mos5_full_sdp/G_300000.pth
lexicon=lexicon/v1.9/bilingual_serving_dict_er3.txt
speaker_id=222

# test_data=testdata/jams/jams_ahm_test100_cn.txt
# output_dir=/nfs1/yi.liu/tts/vits/exp/2023Q1/jams_v0.1/0317/tuning_jams_rx_full_sdp/jams_ahm_test100_cn

test_data=testdata/jams/jams_test_ssmlv2.txt
output_dir=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0419/tuning_jams_mos5_full_sdp/jams_test_ssmlv2

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=7 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir
