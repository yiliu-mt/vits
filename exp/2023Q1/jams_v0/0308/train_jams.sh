# python tools/check_training_config.py -c exp/2023Q1/jams_v0/0308/configs/train_jams_rx/train.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
#     -c exp/2023Q1/jams_v0/0308/configs/train_jams_rx/train.json \
#     -m exp/2023Q1/jams_v0/0308/train_jams_rx \

CUDA_VISIBLE_DEVICES=4,5,6,7 python train_ms.py \
    -c exp/2023Q1/jams_v0/0308/configs/train_jams_rx/train_full.json \
    -m exp/2023Q1/jams_v0/0308/train_jams_rx_full \
    -p exp/2023Q1/jams_v0/0308/train_jams_rx
