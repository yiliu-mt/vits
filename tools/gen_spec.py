import argparse
import os
import json
from multiprocessing import Process
import torch
from utils import HParams, load_wav_to_torch
from mel_processing import spectrogram_torch


def load_filepaths(filelist, split="|"):
    with open(filelist, encoding='utf-8') as f:
        filepaths = [line.strip().split(split)[0] for line in f]
    return filepaths


def do_spectrogram(hps, audiopaths):
    import pdb
    pdb.set_trace()
    for audiopath in audiopaths:
        audio, sampling_rate = load_wav_to_torch(audiopath)
        if sampling_rate != hps.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, hps.sampling_rate))
        audio_norm = audio / hps.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = audiopath.replace(".wav", ".spec.pt")
        if not os.path.exists(spec_filename):
            spec = spectrogram_torch(audio_norm, hps.filter_length,
                hps.sampling_rate, hps.hop_length, hps.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)


def prepare_spec(nj, hps, filelist):
    audiopaths = load_filepaths(filelist)
    split_list = [[] for _ in range(nj)]
    for i, audiopath in enumerate(audiopaths):
        split_list[i % nj].append(audiopath)

    i = 0
    do_spectrogram(hps, split_list[i])

    processes = [Process(
        target=do_spectrogram, args=(hps, split_list[i])
    ) for i in range(nj)]
    for proc in processes:
        proc.daemon = True
        proc.start()
    for proc in processes:
        proc.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nj", type=int, default=16, help="num of workers")
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)

    prepare_spec(args.nj, hparams, hparams.data.training_files)
    prepare_spec(args.nj, hparams, hparams.data.validation_files)


if __name__ == "__main__":
    main()