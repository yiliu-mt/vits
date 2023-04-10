# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /data:/data -v /root/data:/root/data -it mycrazycracy/vits:v1.0 bash

# train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ms.py \
    -c exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train_full_sdp.json \
    -m exp/2023Q1/jams_v0.2/0317/tuning_jams_rx_full_sdp \
    -p exp/2023Q1/jams_v0.2/0317/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_sdp
