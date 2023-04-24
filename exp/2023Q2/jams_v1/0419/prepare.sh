# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $HOME:/local -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v $PWD:/workspace/vits -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it mycrazycracy/vits:v1.0 bash

cut -d '|' -f 1 /local/jams_v1.3_0418/preprocessed_data/haitian/train_val_mos5.txt > filelist
grep -f filelist /local/jams_v1.3_0418/vits/preprocessed_data/haitian/train_val.txt > /local/jams_v1.3_0418/vits/preprocessed_data/haitian/train_val_mos5.txt
