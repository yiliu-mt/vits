# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /root/data:/root/data -it mycrazycracy/vits:v1.0 bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/vits_v0.1/0307/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train.json \
    -m exp/2023Q1/vits_v0.1/0307/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
