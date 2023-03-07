docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it sh-harbor.mthreads.com/mt-ai/vits:v1.0 bash

python tools/generate_vits_data.py \
    --train_txt_fpath train.txt \
    --val_txt_fpath val.txt \
    --remove_silence true \
    --use_sid \
    -p exp/aishell3/1226/configs/preprocess.yaml \
    -o exp/aishell3/1226/preprocessed_data

python gen_spec.py exp/aishell3/1226/configs/aishell3_trim.json