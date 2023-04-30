# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -it mycrazycracy/vits:v1.0 bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q2/jams_v1/0428/configs/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/train_sdp_prosody.json \
    -m exp/2023Q2/jams_v1/0428/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/train_sdp_prosody
