
# python tools/check_training_config.py -c exp/2023Q1/jams_v0/0310/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/tuning_jams_rx/train.json

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/jams_v0/0310/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/tuning_jams_rx/train.json \
    -m exp/2023Q1/jams_v0/0310/tuning_jams_rx \
    -p exp/2023Q1/jams_v0/0310/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022