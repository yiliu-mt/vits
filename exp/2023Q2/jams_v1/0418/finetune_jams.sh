# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME:/local -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q2/jams_v1/0418/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams/train_full_sdp.json \
    -m exp/2023Q2/jams_v1/0418/tuning_jams_full_sdp \
    -p exp/2023Q1/jams_v0.1/0317/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_sdp
