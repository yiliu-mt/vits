# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

# generate jams training data
export PYTHONPATH=.
python tools/generate_vits_data.py \
    --train_txt_fpath train_val.txt \
    --test_txt_fpath test.txt \
    --remove_silence true \
    --use_sid \
    -p /nfs1/yi.liu/src/TTS-FastSpeech/exp/2023Q2/jams_v1.3/0418/config/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_from_10w/preprocess.yaml \
    -o /data3/jams_v1.3_0418/vits/preprocessed_data/haitian

# change the training and val list in the config
python gen_spec.py exp/2023Q2/jams_v1/0418/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams/train_full_sdp.json
