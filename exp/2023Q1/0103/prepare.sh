docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /data:/data -it mycrazycracy/wetts:1.0 bash

# with jams
python tools/generate_vits_data.py \
    --train_txt_fpath train.txt \
    --val_txt_fpath val.txt \
    --test_txt_fpath test.txt \
    --remove_silence true \
    --use_sid \
    -p exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/preprocess.yaml \
    -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228
python gen_spec.py exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228.json
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228


# without jams
python tools/generate_vits_data.py \
    --train_txt_fpath train.txt \
    --val_txt_fpath val.txt \
    --test_txt_fpath test.txt \
    --remove_silence true \
    --use_sid \
    -p exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/preprocess.yaml \
    -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
python gen_spec.py exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

