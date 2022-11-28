#!/usr/bin/env python3

# Preprocess the data to accelerate training
# Refer to wenet for more details

import argparse
import os
import io
import logging
import json
import multiprocessing
import tarfile
import sox
import time
import torch
import torchaudio
import torchaudio.backend.sox_io_backend as sox
from mel_processing import spectrogram_torch

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def load_filepaths_and_text(filename, split="|"):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            info = line.strip().split(split)
            uttid = info[0].split('/')[-1].rsplit('.', 1)[0]
            if len(info) == 2:
                filepaths, text = info
                data.append((uttid, text, filepaths))
            else:
                filepaths, sid, text = info[1], info[2]
                data.append((uttid, text, sid, filepaths))
    return data


def write_tar_file(config,
                   data_list,
                   tar_file,
                   resample,
                   index=0,
                   total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index+1, total))
    read_time = 0.0
    save_time = 0.0
    spec_time = 0.0
    write_time = 0.0
    with tarfile.open(tar_file, "w") as tar:
        for item in data_list:
            if len(item) == 3:
                key, txt, wav = item
                sid = None
            else:
                key, txt, sid, wav = item

            suffix = wav.split('.')[-1]
            assert suffix in AUDIO_FORMAT_SETS
            ts = time.time()
            audio, sample_rate = sox.load(wav, normalize=False)
            read_time += (time.time() - ts)

            # resample
            if sample_rate != resample:
                print(f"Resample {key} from sr:{sample_rate} to sr:{resample}")
                if not audio.is_floating_point():
                    # normalize the audio before resample
                    # because resample can't process int audio
                    audio = audio / (1 << 15)
                    audio = torchaudio.transforms.Resample(
                        sample_rate, resample)(audio)
                    audio = (audio * (1 << 15)).short()
                else:
                    audio = torchaudio.transforms.Resample(
                        sample_rate, resample)(audio)

            ts = time.time()
            f = io.BytesIO()
            sox.save(f, audio, resample, format="wav", bits_per_sample=16)
            suffix = "wav"
            f.seek(0)
            data = f.read()
            save_time += (time.time() - ts)

            ts = time.time()
            audio = audio[0]  # Get the first channel
            audio_norm = audio / config['data']['max_wav_value']
            audio_norm = audio_norm.unsqueeze(0)
            spec = spectrogram_torch(audio_norm,
                                     config['data']['filter_length'],
                                     config['data']['sampling_rate'],
                                     config['data']['hop_length'],
                                     config['data']['win_length'],
                                     center=False)
            spec = torch.squeeze(spec, 0)
            fs = io.BytesIO()
            torch.save(spec, fs)
            fs.seek(0)
            spec_data = fs.read()
            spec_time += (time.time() - ts)

            assert isinstance(txt, str)
            ts = time.time()
            txt_file = key + '.txt'
            txt = txt.encode('utf8')
            txt_data = io.BytesIO(txt)
            txt_info = tarfile.TarInfo(txt_file)
            txt_info.size = len(txt)
            tar.addfile(txt_info, txt_data)

            if sid is not None:
                sid_file = key + '.sid'
                sid = sid.encode('utf-8')
                sid_data = io.BytesIO(sid)
                sid_info = tarfile.TarInfo(sid_file)
                sid_info.size = len(sid)
                tar.addfile(sid_info, sid_data)

            wav_file = key + '.' + suffix
            wav_data = io.BytesIO(data)
            wav_info = tarfile.TarInfo(wav_file)
            wav_info.size = len(data)
            tar.addfile(wav_info, wav_data)

            spec_file = key + '.spec'
            spec_bytes = io.BytesIO(spec_data)
            spec_info = tarfile.TarInfo(spec_file)
            spec_info.size = len(spec_data)
            tar.addfile(spec_info, spec_bytes)

            write_time += (time.time() - ts)
        logging.info('read {} save {} spec {} write {}'.format(read_time, save_time, spec_time, write_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config",
                        help='JSON file for configuration')
    parser.add_argument('--num_threads',
                        type=int,
                        default=1,
                        help='num threads for make shards')
    parser.add_argument('--num_utts_per_shard',
                        type=int,
                        default=1000,
                        help='num utts per shard')
    parser.add_argument('--prefix',
                        default='shards',
                        help='prefix of shards tar file')
    parser.add_argument('filelist', help='The input file list')
    parser.add_argument('shards_dir', help='output shards dir')
    parser.add_argument('shards_list', help='output shards list file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch.set_num_threads(1)
    data = load_filepaths_and_text(args.filelist)

    num = args.num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.shards_dir, exist_ok=True)

    config_path = args.config
    config_save_path = os.path.join(args.shards_dir, "config.json")
    with open(config_path, "r") as f:
        data = f.read()
    with open(config_save_path, "w") as f:
        f.write(data)
    config = json.loads(data)

    # Using thread pool to speedup
    pool = multiprocessing.Pool(processes=args.num_threads)
    shards_list = []
    tasks_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(args.shards_dir,
                                '{}_{:09d}.tar'.format(args.prefix, i))
        shards_list.append(tar_file)

        pool.apply_async(
            write_tar_file,
            (config, chunk, tar_file, config['data']['sampling_rate'], i, num_chunks))
    pool.close()
    pool.join()

    with open(args.shards_list, 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')
