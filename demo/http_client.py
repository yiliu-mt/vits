import argparse
import sys
from collections import defaultdict
import time
import requests
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_scale", type=float, default=0.667)
    parser.add_argument("--noise_scale_w", type=float, default=0.8)
    parser.add_argument("service_address", type=str, default="127.0.0.1:8911", help="service address")
    parser.add_argument("text", type=str, default="")
    args = parser.parse_args()

    # debug
    args.service_address = "172.31.208.11:9000"
    args.text = "<speak>首先，<phoneme ph='wo2 xiang3'>我想</phoneme>借此机会，感谢<phoneme ph='yi2 lu4 zou3 lai2'>一路走来</phoneme><break time='sp0'></break><phoneme ph='bu2 duan4'>不断</phoneme>支持摩尔线程的合作伙伴、开发者<break time='sp0'></break>以及玩家，</speak>"


    api_address = "http://{}/generate".format(args.service_address)

    speaker_metrics_list = defaultdict(list)

    text = args.text.strip()

    request_payload = {
        "text": text,
        "format": "wav",
        "noise_scale": args.noise_scale,
        "noise_scale_w": args.noise_scale_w,
    }

    start = time.time()
    resp = requests.post(api_address, json=request_payload)
    assert resp.status_code == 200, "{}".format(text)
    print("Success!!!")
    tc = time.time() - start

    with open("demo.wav", "wb") as f:
        f.write(resp.content)
    import wave
    with wave.open('demo.wav', 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        print(f"Duration: {duration:.2f} seconds")
