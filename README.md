# mThreads VITS

This repo is used to train and do inference for mThreads VITS.

## Usage

###  Docker

```
cd ${path_of_the_code}
docker run --rm --gpus all --ipc=host -v $PWD:/workspace/vits -v /nfs1:/nfs1 -v /nfs2:/nfs2 -it sh-harbor.mthreads.com/mt-ai/vits:musa bash
```

### Training

```
# Get the pretrained model
mkdir -p logs
ln -s /nfs1/yi.liu/tts/vits/exp/2023Q1/jams_v0/0308/Baker_LJSpeech_MuSha0914_RxEnhancedV5_AISHELL3_Mita1022/ logs/musa_pretrained

# Get the training data
ln -s /nfs1/yi.liu/src/vits/preprocessed_data preprocessed_data

# Finetune from the pretrained model
bash exp/2023Q1/musa/train.sh
```

### Inference

```
bash exp/2023Q1/musa/synthesize.sh

```
