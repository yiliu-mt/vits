
python tools/generate_vits_data.py \
    --train_txt_fpath train.txt \
    --val_txt_fpath val.txt \
    --test_txt_fpath test.txt \
    --remove_silence true \
    -p exp/baker/1221/configs/preprocess.yaml \
    -o exp/baker/1221/preprocessed_data
