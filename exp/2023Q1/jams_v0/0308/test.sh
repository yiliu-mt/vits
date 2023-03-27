# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

config=exp/2023Q1/jams_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test.json
# config=logs/exp/2023Q1/jams_v0/0308/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/config.json
model=logs/exp/2023Q1/jams_v0/0308/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/G_400000.pth
lexicon=lexicon/v1.9/bilingual_serving_dict_er3.txt
speaker_id=800

test_data=testdata/jams/new_year.txt
output_dir=exp/2023Q1/jams_v0/0308/audio_out/new_year

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir
