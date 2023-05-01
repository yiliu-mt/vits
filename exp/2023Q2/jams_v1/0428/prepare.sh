# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

export PYTHONPATH=.
python tools/generate_vits_data.py \
    --train_txt_fpath train_val.txt \
    --test_txt_fpath val.txt \
    --remove_silence true \
    --use_sid \
    -p /nfs1/yi.liu/src/TTS-FastSpeech/exp/2023Q2/jams_v1.3/0428/config/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocess.yaml \
    -o /data3/jams_v1.3_0428_prosody/vits/preprocessed_data

# Now the spN in /data3/jams_v1.3_0428_prosody/vits/preprocessed_data is wrong!
cd /nfs1/yi.liu/src/datasets/jams/prosody_process_20230427
export PYTHONPATH=.
python utils/generate_vits_prosody_text.py --type "insert" --lexicon merged_dict_all.txt \
    --reference /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/train_val.txt \
    /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val.txt \
    /data2/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/text_predict_prosody_20230427.txt \
    /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt
cd -

# Fix spN
mv /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt.bak
python tools/fix_spn.py \
    --reference /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/train_val.txt \
    --input /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt.bak \
    --output /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt

cut -d '|' -f 1 /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/train.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train.list
cut -d '|' -f 1 /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/val.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val.list
cut -d '|' -f 1 /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/train_jams_all.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_all.list
cut -d '|' -f 1 /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/train_jams_haitian.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian.list

# train_prosody.txt
# val_prosody.txt
# train_jams_all_prosody.txt
# train_jams_haitian_prosody.txt
# val_jams_prosody.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_prosody.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val_prosody.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_all.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_all_prosody.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian_prosody.txt
head -n 3 /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian_prosody.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val_jams_prosody.txt


# Prepare the list for fastspeech-like training
python tools/combine_fastspeech_prosody.py \
    --reference /data3/jams_v1.3_0428_prosody/Baker_LJSpeech_MuSha0914_AISHELL3_Mita1022_Jams0423/preprocessed_data/train_val.txt \
    --input /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val.txt \
    --output /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody_fastspeech.txt

# train_prosody_fastspeech.txt
# val_prosody_fastspeech.txt
# train_jams_all_prosody_fastspeech.txt
# train_jams_haitian_prosody_fastspeech.txt
# val_jams_prosody_fastspeech.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody_fastspeech.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_prosody_fastspeech.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody_fastspeech.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val_prosody_fastspeech.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_all.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody_fastspeech.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_all_prosody_fastspeech.txt
grep -f /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian.list /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_val_prosody_fastspeech.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian_prosody_fastspeech.txt
head -n 3 /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/train_jams_haitian_prosody_fastspeech.txt > /data3/jams_v1.3_0428_prosody/vits/preprocessed_data/val_jams_prosody_fastspeech.txt
