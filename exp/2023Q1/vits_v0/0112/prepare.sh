CUDA_VISIBLE_DEVICES=4,5,6,7 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022.json -m exp/2023Q1/0112/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

# old model
# can only finetuned from 200K steps
# CUDA_VISIBLE_DEVICES=0,1 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
CUDA_VISIBLE_DEVICES=0,1 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json -m exp/2023Q1/0103/tuning_mita_bc_chatbot10_190 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10

# CUDA_VISIBLE_DEVICES=2,3 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
CUDA_VISIBLE_DEVICES=2,3 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10.json -m exp/2023Q1/0103/tuning_musha_mt10_190 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10


# train from 300k
CUDA_VISIBLE_DEVICES=0,1 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json -m exp/2023Q1/0112/tuning_mita_bc_chatbot10_300k -p exp/2023Q1/0112/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
CUDA_VISIBLE_DEVICES=2,3 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10.json -m exp/2023Q1/0112/tuning_musha_mt10_300k -p exp/2023Q1/0112/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

# train from 400k
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json -m exp/2023Q1/0112/tuning_mita_bc_chatbot10 -p exp/2023Q1/0112/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_ms.py -c exp/2023Q1/0112/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10.json -m exp/2023Q1/0112/tuning_musha_mt10 -p exp/2023Q1/0112/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022