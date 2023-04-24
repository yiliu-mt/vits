# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

export PYTHONPATH=.
python tools/generate_vits_data.py \
    --train_txt_fpath train.txt \
    --test_txt_fpath val.txt \
    --remove_silence true \
    --use_sid \
    -p /nfs1/yi.liu/src/TTS-FastSpeech/exp/2023Q2/jams_v1.3/0423/config/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocess.yaml \
    -o /data3/jams_v1.3_0423/vits/preprocessed_data
