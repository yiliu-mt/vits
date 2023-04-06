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
bash exp/2023Q1/musa/train.sh
```

### Inference

```
bash exp/2023Q1/musa/synthesize.sh

```
