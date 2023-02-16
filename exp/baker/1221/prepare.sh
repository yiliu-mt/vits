
python tools/generate_vits_data.py \
    --train_txt_fpath train.txt \
    --val_txt_fpath val.txt \
    --test_txt_fpath test.txt \
    --remove_silence true \
    -p exp/baker/1221/configs/preprocess.yaml \
    -o exp/baker/1221/preprocessed_data

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it sh-harbor.mthreads.com/mt-ai/vits:v1.0 bash