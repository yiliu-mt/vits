# docker run --rm --gpus all -p 8911:8911 -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vits:demo bash

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run python gradio/run.py \
    --config /nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0418/tuning_jams_full_sdp/config.json \
    --model /nfs1/yi.liu/tts/vits/exp/2023Q2/jams_v1/0418/tuning_jams_full_sdp/G_200000.pth \
    --lexicon lexicon/v1.9/bilingual_serving_dict_er3.txt --speaker_id 222