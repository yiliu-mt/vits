# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu/:/home/ubuntu -it mycrazycracy/vits:v1.0 bash

# train from mita 200K model
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/vits_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10_full/train.json \
    -m exp/2023Q1/vits_v0/0308/tuning_mita_bc_chatbot10_full \
    -p /workspace/vits/preprocessed_data/tuning_mita_bc_chatbot10
