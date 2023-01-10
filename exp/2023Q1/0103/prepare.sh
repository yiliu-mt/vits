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


# finetune model
# prepare lists
# musha
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/musha/train_musha_mt10.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train_musha_mt10.txt
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/musha/musha_test.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/all.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_musha.txt

python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/musha/train_musha_mt10.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_musha_mt10.txt
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/musha/musha_test.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/all.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/test_musha.txt

# mita
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/mita/train_chatbot10.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train_mita_chatbot10.txt
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/mita/test.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/all.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_mita.txt

python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/mita/train_chatbot10.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_mita_chatbot10.txt
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/mita/test.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/all.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/test_mita.txt

# jams
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/jams/train_all.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/train_jams.txt
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/jams/val.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/test_jams.txt

python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/jams/train_all.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_jams.txt
python tools/make_list_from_id.py -r exp/2023Q1/0103/lists/jams/val.txt -i exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/train_val.txt -o exp/2023Q1/0103/preprocessed_data/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/test_jams.txt

# train models
CUDA_VISIBLE_DEVICES=0,1 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_mita_bc_chatbot10 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
CUDA_VISIBLE_DEVICES=2,3 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_musha_mt10 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022
# CUDA_VISIBLE_DEVICES=2,3 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/tuning_jams_rx -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022

CUDA_VISIBLE_DEVICES=4,5 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/tuning_mita_bc_chatbot10.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/tuning_mita_bc_chatbot10 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228
CUDA_VISIBLE_DEVICES=6,7 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/tuning_musha_mt10.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/tuning_musha_mt10 -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228
# CUDA_VISIBLE_DEVICES=6,7 python train_ms.py -c exp/2023Q1/0103/configs/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/tuning_jams_rx.json -m Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228/tuning_jams_rx -p Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022_Jams1228
