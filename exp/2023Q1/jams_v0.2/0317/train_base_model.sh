# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -it mycrazycracy/vits:v1.0 bash

# python tools/check_training_config.py -c exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
CUDA_VISIBLE_DEVICES=1 python train_ms.py \
    -c exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train.json \
    -m exp/2023Q1/jams_v0.2/0317/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022