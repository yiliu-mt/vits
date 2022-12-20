import argparse
import os
import json
import math
import time
import torch
import numpy as np
from scipy.io import wavfile

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file")
    parser.add_argument("-t", "--test_file", help="test file")
    parser.add_argument("-m", "--model", help="model file")
    parser.add_argument("-o", "--output_dir", help="output directory")
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.model, net_g, None)

    # do actual work
    audiopaths_and_text = utils.load_filepaths_and_text(args.test_file)
    with torch.no_grad():
        for audiopath, text in audiopaths_and_text:
            print(audiopath)
            text_norm = cleaned_text_to_sequence(text)
            if hps.data.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            text_padded = torch.LongTensor(text_norm)
            x_tst = text_padded.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([text_padded.size(0)]).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
            audio = np.clip(audio, -32767.0, 32767.0)
            wav_name = audiopath.split("/")[-1]
            wavfile.write(os.path.join(args.output_dir, wav_name),
                          hps.data.sampling_rate, audio.astype(np.int16))


if __name__ == "__main__":
    main()


# def get_text(text, hps):
#     text_norm = text_to_sequence(text, hps.data.text_cleaners)
#     if hps.data.add_blank:
#         text_norm = commons.intersperse(text_norm, 0)
#     text_norm = torch.LongTensor(text_norm)
#     return text_norm
# 
# stn_tst = get_text("This is a sample audio", hps)
# with torch.no_grad():
#     x_tst = stn_tst.cuda().unsqueeze(0)
#     x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
#     audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

# # warm-up
# print("Warming up...")
# audiopaths_and_text = utils.load_filepaths_and_text('filelists/warmup.txt')
# with torch.no_grad():
#     for audiopath, text in audiopaths_and_text:
#         text_norm = cleaned_text_to_sequence(text)
#         if hps.data.add_blank:
#             text_norm = commons.intersperse(text_norm, 0)
#         text_padded = torch.LongTensor(text_norm)
#         x_tst = text_padded.to(device).unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([text_padded.size(0)]).to(device)
#         audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()