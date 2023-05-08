# docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -v /nfs1/yichao.hu:/data -v /home/yichao.hu/nltk_data:/root/nltk_data -v /nfs2/yi.liu/data:/data2 -v /nfs1/yi.liu/tts:/data3 -it sh-harbor.mthreads.com/mt-ai/vits:v1.0 bash

config=/nfs1/yi.liu/tts/vits/exp/2023Q1/0308/tuning_mita_bc_chatbot10_full/config.json
model=/nfs1/yi.liu/tts/vits/exp/2023Q1/0308/tuning_mita_bc_chatbot10_full/G_260000.pth
lexicon=/nfs1/yi.liu/src/TTS-FastSpeech/lexicon/v1.9/bilingual_serving_dict_er3.txt
speaker_id=221

test_data=exp/2023Q1/vits_v0/0308/gwsk.txt
output_dir=audio_outs

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python synthesize_v2.py \
    -c $config \
    -m $model \
    -t $test_data \
    --lexicon $lexicon \
    -s $speaker_id \
    -o $output_dir

