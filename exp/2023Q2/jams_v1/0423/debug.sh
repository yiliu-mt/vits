# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

config=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0419/tuning_jams_mos5_full_sdp/config.json
model=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0419/tuning_jams_mos5_full_sdp/G_300000.pth
lexicon=lexicon/v1.9/bilingual_serving_dict_er3.txt
speaker_id=222

export PYTHONPATH=.

# test_data=exp/2023Q2/jams_v1/0423/train_list.txt
# output_dir=audio_out
# CUDA_VISIBLE_DEVICES=7 python synthesize_debug.py \
#     -c $config \
#     -m $model \
#     -t $test_data \
#     --lexicon $lexicon \
#     -s $speaker_id \
#     -o $output_dir

test_data=exp/2023Q2/jams_v1/0423/test.txt
output_dir=audio_out2
CUDA_VISIBLE_DEVICES=7 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir
