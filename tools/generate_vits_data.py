import argparse
import os
import logging
import json
import yaml
import librosa
import tgt
import torch
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.io import wavfile
from utils import load_wav_to_torch, combine_list
from mel_processing import spectrogram_torch
from multiprocessing import Process


sil_phones = ["sil", "sp", "spn", "sp1", "sp2", "sp3", "sp0", "spb"]


def stats_sil_phones_from_path(path):
    spk2sos_durations = defaultdict(list)
    spk2eos_durations = defaultdict(list)
    spk2sil_durations = defaultdict(list)
    spk_dirs = [
        speaker for speaker in os.listdir(os.path.join(path, "TextGrid"))
        if os.path.isdir(os.path.join(path, "TextGrid", speaker))
    ]

    for speaker in spk_dirs:
        for tg_name in os.listdir(os.path.join(path, "TextGrid", speaker)):
            if ".TextGrid" not in tg_name:
                continue
            tg_path = os.path.join(path, "TextGrid", speaker, tg_name)
            textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)

            for p_idx, t in enumerate(textgrid.get_tier_by_name("phones")._objects):
                s, e, p = t.start_time, t.end_time, t.text
                if p == "":
                    p = "sp"
                # For silent phones
                if p in sil_phones:
                    if p_idx == 0:
                        spk2sos_durations[speaker].append(t.end_time - t.start_time)
                    elif p_idx == len(textgrid.get_tier_by_name("phones")._objects) - 1:
                        spk2eos_durations[speaker].append(t.end_time - t.start_time)
                    else:
                        spk2sil_durations[speaker].append(t.end_time - t.start_time)
    return spk2sos_durations, spk2eos_durations, spk2sil_durations


def get_phones(between_words, sample_rate, hop_length, spk2pause_stats, tg_name):
    '''
    Output the phone sequence and the duration of each phone.
    Also the actual start and end time of speech is marked.
    NOTE: 
    The duration depends on the parameter in "preprocess_config".
    If we change the config (sample rate, frame length, hop length, etc.), the duration should be 
    generated again!

    Args:
        between_words: If true, we manually add blank between words (not phones)
        preprocess_config: The config is similar to the one used in FastSpeech
        spk2pause_stats: help us to convert silence into different levels
        tg_name: the input TextGrid file
    
    Returns:
        phones: the converted phone sequence
        durations: the duration of each phone
        start_time, end_time: the start and end time of the speech
    '''
    textgrid = tgt.io.read_textgrid(tg_name, include_empty_intervals=True)

    # map the words and the corresponding phones
    tier_words, tier_phones = textgrid.get_tier_by_name("words"), textgrid.get_tier_by_name("phones")
    word2phones = []
    for index, word in enumerate(tier_words._objects):
        sw, ew, w = word.start_time, word.end_time, word.text
        word2phones.append([w, []])
        for phone in tier_phones._objects:
            sp, ep, p = phone.start_time, phone.end_time, phone.text
            if ep <= sw:
                continue
            if sp >= ew:
                break
            word2phones[index][1].append([sp, ep, p])
    
    assert len(tier_phones) == sum([len(word_phone[1]) for word_phone in word2phones]), "Error in parsing the word and phone info"

    sp1_threshold, sp2_threshold, sp3_threshold = spk2pause_stats

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for word_index, word_phone in enumerate(word2phones):
        is_silence = False
        
        for phone in word_phone[1]:
            s, e, p = phone 

            if p == "" or p in sil_phones:
                is_silence = True
                duration = e - s
                if duration < sp1_threshold:
                    p = "sp0"
                elif duration < sp2_threshold:
                    p = "sp1"
                elif duration < sp3_threshold:
                    p = "sp2"
                else:
                    p = "sp3"
        
            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * sample_rate / hop_length)
                    - np.round(s * sample_rate / hop_length)
                )
            )

        if between_words:
            if word_index == 0 and is_silence:
                continue
            if word_index == len(word2phones) -1 and is_silence:
                continue
            phones.append("#0")
            end_idx = len(phones)

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    return phones, durations, start_time, end_time


def output_phones(out, input_file, output_file):
    with open(input_file) as fp_in, open(output_file, 'w') as fp_out:
        for line in fp_in:
            name = line.strip().split("|")[0]
            if name not in out:
                continue

            if len(out[name]) == 3:
                # without sid
                fp_out.write('{}|{}|{}\n'.format(
                    out[name][0],
                    ' '.join(out[name][1]),
                    ' '.join([str(d) for d in out[name][2]])
                ))
            else:
                # with sid
                fp_out.write('{}|{}|{}|{}\n'.format(
                    out[name][0],
                    out[name][2],
                    ' '.join(out[name][1]),
                    ' '.join([str(d) for d in out[name][3]])
                ))


def check_list(input_text, output_text):
    wav_set = set()
    with open(output_text) as f:
        for line in f:
            name = line.strip().split("|")[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            wav_set.add(name)
    with open(input_text) as f:
        for line in f:
            name = line.strip().split("|")[0]
            if name not in wav_set:
                logging.warning(f"Skip {name} since it is not found.")


def do_work(
    index,
    args,
    preprocess_config,
    config,
    spk2pause_stats,
    spk2id,
    spk_list,
    output_wav_dir,
    output_list_dir
):
    raw_path = preprocess_config["path"]["raw_path"]
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]

    # Load the STFT parameters
    max_wav_value = config["data"]["max_wav_value"] if config is not None else \
        preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    sample_rate = config["data"]["sampling_rate"] if config is not None else \
        preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["data"]["hop_length"] if config is not None else \
        preprocess_config["preprocessing"]["stft"]["hop_length"]
    win_length = config["data"]["win_length"] if config is not None else \
        preprocess_config["preprocessing"]["stft"]["win_length"]
    filter_length = config["data"]["filter_length"] if config is not None else \
        preprocess_config["preprocessing"]["stft"]["filter_length"]
    
    out = {}
    for speaker in spk_list:
        for tg_name in os.listdir(os.path.join(preprocessed_path, "TextGrid", speaker)):
            if ".TextGrid" not in tg_name:
                continue
            basename = tg_name.rsplit(".", 1)[0]
            wav_path = os.path.join(raw_path, speaker, "{}.wav".format(basename))

            phones, duration, start, end = get_phones(
                args.words,
                sample_rate,
                hop_length,
                spk2pause_stats[speaker],
                os.path.join(preprocessed_path, "TextGrid", speaker, tg_name)
            )

            sr, wav = wavfile.read(wav_path)
            assert sr == sample_rate

            if args.remove_silence:
                # trim wav
                start_time = int(sample_rate * start)
                end_time = start_time + int(sum(duration) * hop_length)
                # Check whether the duration is correct. The total duration should be close to the actual end time.
                assert abs(end_time / sample_rate - end) < (hop_length / sample_rate), \
                    "Duration wrong {} vs {}? {}".format(end_time / sample_rate, end, wav_path)
                if end_time > wav.shape[0]:
                    logging.warning("The length of transcription is larger than the wav: {}".format(wav_path))
                    end_time = wav.shape[0]
                    start_time = max(end_time - int(sum(duration) * hop_length), 0)
                wav = wav[start_time:end_time]
            else:
                # add silence to the label
                # phones = ['sil'] + phones + ['sil']
                raise NotImplementedError("Remove silence before training")

            # write wave
            os.makedirs(os.path.join(output_wav_dir, "Wave", speaker), exist_ok=True)
            out_wav_path = os.path.join(output_wav_dir, "Wave", speaker, "{}.wav".format(basename))
            wavfile.write(out_wav_path, sr, wav.astype(np.int16))

            # Extract spectrogram
            audio_norm = torch.from_numpy(wav) / max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            spec_filename = out_wav_path.replace(".wav", ".spec.pt")
            spec = spectrogram_torch(audio_norm, filter_length,
                sample_rate, hop_length, win_length,
                center=False)
            assert spec.size(2) == sum(duration), "Mismatch: spec {} vs duration {}".format(spec.size(2), sum(duration))
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

            if args.use_sid:
                out[basename] = [out_wav_path, phones, spk2id[speaker], duration]
            else:
                out[basename] = [out_wav_path, phones, duration]
    
    output_phones(
        out,
        os.path.join(preprocessed_path, args.train_txt_fpath),
        os.path.join(output_list_dir, '{}.{}'.format(args.train_txt_fpath, index))
    )
    if args.val_txt_fpath is not None:
        output_phones(
            out,
            os.path.join(preprocessed_path, args.val_txt_fpath),
            os.path.join(output_list_dir, '{}.{}'.format(args.val_txt_fpath, index))
        )
    if args.test_txt_fpath is not None:
        output_phones(
            out,
            os.path.join(preprocessed_path, args.test_txt_fpath),
            os.path.join(output_list_dir, '{}.{}'.format(args.test_txt_fpath, index))
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nj", type=int, default=32, help="num of workers")
    parser.add_argument("-w", "--words", action="store_true", help="add #0 between words")
    parser.add_argument("--train_txt_fpath", type=str, required=True, default="train.txt")
    parser.add_argument("--val_txt_fpath", type=str, default=None)
    parser.add_argument("--test_txt_fpath", type=str, default=None)
    parser.add_argument("--use_sid", action="store_true", help="add speaker id to the labels")
    parser.add_argument(
        "--remove_silence",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="To remove the heading and tailing silence"
    )
    parser.add_argument("-p", "--preprocess_config", required=True, type=str)
    parser.add_argument("-o", "--output_dir", required=True, type=str)
    parser.add_argument("-c", "--config", default=None, type=str, help="The config used in vits training")
    parser.add_argument("-l", "--output_list_dir", default=None, type=str, help="The output dir of the list")
    args = parser.parse_args()

    # Load the speaker info
    config = None
    if args.config is not None:
        with open(args.config, "r") as f:
            data = f.read()
        config = json.loads(data)

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    output_wav_dir = args.output_dir
    output_list_dir = args.output_list_dir if args.output_list_dir is not None else output_wav_dir
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(output_list_dir, exist_ok=True)

    spk_dirs = [
        speaker for speaker in os.listdir(os.path.join(preprocessed_path, "TextGrid"))
        if os.path.isdir(os.path.join(preprocessed_path, "TextGrid", speaker))
    ]
    # Sort the speakers by the names
    spk_dirs = sorted(spk_dirs)
    print("all spks {}".format(spk_dirs))
    print("Total num of spks: {}".format(len(spk_dirs)))

    if args.use_sid:
        if "speakers_path" in preprocess_config["path"]:
            spk2id = json.load(open(preprocess_config["path"]["speakers_path"]))
        elif os.path.exists(os.path.join(preprocessed_path, "speakers.json")):
            spk2id = json.load(open(os.path.join(preprocessed_path, "speakers.json")))
        else:
            spk2id = {}
            for sid, speaker in enumerate(spk_dirs):
                spk2id[speaker] = sid
    print(spk2id)

    # Compute the stats of silence of each speaker
    sos_sil_stat, eos_sil_stat, sil_stat = stats_sil_phones_from_path(preprocess_config["path"]["preprocessed_path"])
    spk2pause_stats = dict()
    for spk in sos_sil_stat.keys():
        print("====={}".format(spk))
        pos_stats = pd.Series(sil_stat[spk]).describe(percentiles=[.3, .6, .8, .99])
        spk2pause_stats[spk] = [pos_stats["30%"], pos_stats["60%"], pos_stats["80%"]]

    if args.use_sid:
        with open(os.path.join(output_list_dir, "speakers.json"), 'w') as f:
            json.dump(spk2id, f)

    split_spk_list = [[] for _ in range(args.nj)]
    for i, speaker in enumerate(spk_dirs):
        split_spk_list[i % args.nj].append(speaker)

    # i = 0
    # do_work(
    #     i, args, preprocess_config, config, spk2pause_stats, spk2id,
    #     split_spk_list[i], output_wav_dir, output_list_dir
    # )
    # quit()

    processes = [Process(
        target=do_work, args=(
            i, args, preprocess_config, config, spk2pause_stats, spk2id,
            split_spk_list[i], output_wav_dir, output_list_dir
        )
    ) for i in range(args.nj)]
    for proc in processes:
        proc.daemon = True
        proc.start()
    for proc in processes:
        proc.join()

    # combine text
    combine_list(args.nj, os.path.join(output_list_dir, args.train_txt_fpath))
    check_list(os.path.join(preprocessed_path, args.train_txt_fpath), os.path.join(output_list_dir, args.train_txt_fpath))
    os.system("rm {}".format(" ".join([os.path.join(output_list_dir, "{}.{}".format(args.train_txt_fpath, i)) for i in range(args.nj)])))
    if args.val_txt_fpath is not None:
        combine_list(args.nj, os.path.join(output_list_dir, args.val_txt_fpath))
        check_list(os.path.join(preprocessed_path, args.val_txt_fpath), os.path.join(output_list_dir, args.val_txt_fpath))
        os.system("rm {}".format(" ".join([os.path.join(output_list_dir, "{}.{}".format(args.val_txt_fpath, i)) for i in range(args.nj)])))
    if args.test_txt_fpath is not None:
        combine_list(args.nj, os.path.join(output_list_dir, args.test_txt_fpath))
        check_list(os.path.join(preprocessed_path, args.test_txt_fpath), os.path.join(output_list_dir, args.test_txt_fpath))
        os.system("rm {}".format(" ".join([os.path.join(output_list_dir, "{}.{}".format(args.test_txt_fpath, i)) for i in range(args.nj)])))


if __name__ == '__main__':
    main()
