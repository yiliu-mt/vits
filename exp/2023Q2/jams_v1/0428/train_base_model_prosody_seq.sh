# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME:/local -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

CUDA_VISIBLE_DEVICES=1,2,4,5 python train_ms.py \
    -c exp/2023Q2/jams_v1/0428/configs/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/train_sdp_prosody_fastspeech.json \
    -m exp/2023Q2/jams_v1/0428/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/train_sdp_prosody_fastspeech
