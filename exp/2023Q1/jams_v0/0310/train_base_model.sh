# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /home/ubuntu:/home/ubuntu -it mycrazycracy/vits:v1.0 bash


# python tools/lists/filter_list.py --include "LJSpeech|SSB3000|SSB3001|SSB3002" \
#     exp/2023Q1/jams_v0/0308/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/speakers.json \
#     exp/2023Q1/jams_v0/0308/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train_val.txt \
#     exp/2023Q1/jams_v0/0310/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/speakers.json \
#     exp/2023Q1/jams_v0/0310/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/train_val.txt
# 
# python tools/lists/filter_list.py --include "LJSpeech|SSB3000|SSB3001|SSB3002" --use-new-speaker-map \
#     exp/2023Q1/jams_v0/0308/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/speakers.json \
#     exp/2023Q1/jams_v0/0308/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/val.txt \
#     exp/2023Q1/jams_v0/0310/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/speakers.json \
#     exp/2023Q1/jams_v0/0310/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/val.txt
# 
# python tools/check_training_config.py -c exp/2023Q1/jams_v0/0310/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/train.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_ms.py \
    -c exp/2023Q1/jams_v0/0310/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022/train.json \
    -m exp/2023Q1/jams_v0/0310/Baker_LJSpeech_MuSha0914_RxEnhancedV5_Mita1022
