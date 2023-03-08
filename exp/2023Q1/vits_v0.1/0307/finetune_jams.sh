# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it mycrazycracy/vits:v1.0 bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/vits_v0.1/0307/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train.json \
    -m exp/2023Q1/vits_v0.1/0307/tuning_jams_rx \
    -p exp/2023Q1/0227/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

-p exp/2023Q1/0112/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022