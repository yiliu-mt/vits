# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME:/local -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 FREEZE_DURATION_PREDICTOR=1 python train_ms.py \
    -c exp/2023Q2/jams_v1/0423/configs/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/tuning_jams_haitian_from_all/train_jams_haitian_full_sdp.json \
    -m exp/2023Q2/jams_v1/0423/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/tuning_jams_all_full_sdp/tuning_jams_haitian_from_all_freeze_dur \
    -p exp/2023Q2/jams_v1/0423/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/tuning_jams_all_full_sdp