wget https://datashare.ed.ac.uk/download/DS_10283_3443.zip

docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it sh-harbor.mthreads.com/mt-ai/vits:v1.0 bash

python gen_spec.py exp/vctk/1226/configs/vctk.json