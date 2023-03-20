# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /data:/data -v /root/data:/root/data -it mycrazycracy/vits:v1.0 bash

# # generate jams training data
# python tools/generate_vits_data.py \
#     --train_txt_fpath train_all.txt \
#     --test_txt_fpath val.txt \
#     --remove_silence true \
#     --use_sid \
#     -p exp/2023Q1/jams_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/preprocess.yaml \
#     -o preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/jams_rx
# python gen_spec.py exp/2023Q1/jams_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train.json

# python tools/check_training_config.py -c exp/2023Q1/jams_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train.json

# # train
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
#     -c exp/2023Q1/jams_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train.json \
#     -m exp/2023Q1/jams_v0/0308/tuning_jams_rx \
#     -p exp/2023Q1/jams_v0/0308/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/jams_v0/0308/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train_full.json \
    -m exp/2023Q1/jams_v0/0308/tuning_jams_rx_part_full \
    -p exp/2023Q1/jams_v0/0308/tuning_jams_rx
