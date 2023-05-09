# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

config=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0423/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/tuning_jams_all_full_sdp/tuning_jams_haitian_from_all/config.json
model=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0423/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/tuning_jams_all_full_sdp/tuning_jams_haitian_from_all/G_200000.pth
lexicon=lexicon/v1.9/bilingual_serving_dict_er3.txt

# In this model, ID=43
speaker_id=43

test_data=testdata/jams/keynote_202305_overall.txt
output_dir=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0423/tuning_jams_all_full_sdp/tuning_jams_haitian_from_all/keynote_202305_overall

# test_data=testdata/jams/test.txt
# output_dir=/nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0423/tuning_jams_all_full_sdp/tuning_jams_haitian_from_all/test

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir
