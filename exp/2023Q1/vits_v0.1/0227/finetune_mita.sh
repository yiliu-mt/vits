
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py \
    -c exp/2023Q1/0227/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10/train.json \
    -p exp/2023Q1/0227/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022 \
    -m exp/2023Q1/0227/tuning_mita_bc_chatbot10