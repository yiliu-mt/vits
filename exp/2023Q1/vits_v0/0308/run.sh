# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it mycrazycracy/vits:v1.0 bash

# train from mita 200K model
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
python train_ms.py \
    -c exp/2023Q1/vits_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10_full/train.json \
    -m exp/2023Q1/vits_v0/0308/tuning_mita_bc_chatbot10_full \
    -p /nfs1/yi.liu/tts/vits/exp/2023Q1/0112/tuning_mita_bc_chatbot10
