# Synthesize with long sentence and SSML support

import argparse
import os
import logging
import numpy as np
import torch
from scipy.io import wavfile
from service import TTS, GenerationRequest, unary_synthesize_text

logging.getLogger().setLevel(logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0, help="speed factor")
    parser.add_argument("-c", "--config", required=True, help="config")
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-t", "--text", required=True, help="text")
    parser.add_argument("-l", "--lexicon", required=True, help="lexicon")
    parser.add_argument("-s", "--speaker_id", type=int, required=True, help="speaker id")
    parser.add_argument("-o", "--output_dir", required=True, help="output_dir")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vits = TTS(args.config, args.model, args.lexicon, device)

    for line in open(args.text):
        utt_id, text = line.strip().split(" ", 1)
        audio_data = unary_synthesize_text(
            vits,
            utt_id,
            GenerationRequest(text=text, format="wav", voice=args.speaker_id, speed_rate=args.speed),
            max_single_utt_length=1
        )
        wavfile.write(
            os.path.join(args.output_dir, "{}.wav".format(utt_id)),
            vits.sample_rate,
            audio_data.astype(np.int16))


if __name__ == "__main__":
    main()




