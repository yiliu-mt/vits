# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /root/data:/root/data -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v `pwd`:/workspace/TTS-FastSpeech -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it mycrazycracy/vits:v1.0 bash

# # generate all training data
# PYTHONPATH=. python tools/generate_vits_data.py \
#     --train_txt_fpath train.txt \
#     --val_txt_fpath val.txt \
#     --test_txt_fpath test.txt \
#     --remove_silence true \
#     --use_sid \
#     -p exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/preprocess.yaml \
#     -c exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train.json \
#     -o preprocessed_data_new/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022 \
#     -l exp/2023Q1/jams_v0.2/0317/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

# generate jams training data
PYTHONPATH=. python tools/generate_vits_data.py \
    --train_txt_fpath train_all.txt \
    --test_txt_fpath val.txt \
    --remove_silence true \
    --use_sid \
    -p exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/preprocess.yaml \
    -c exp/2023Q1/jams_v0.2/0317/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx/train.json \
    -o preprocessed_data_new/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/jams_rx \
    -l exp/2023Q1/jams_v0.2/0317/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/jams_rx