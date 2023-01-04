import argparse
import os
import json
import yaml
import librosa
import tgt
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.io import wavfile


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


def get_phones(between_words, preprocess_config, spk2pause_stats, tg_name):
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
                    np.round(
                        e * preprocess_config["preprocessing"]["audio"]["sampling_rate"] /
                        preprocess_config["preprocessing"]["stft"]["hop_length"])
                    - np.round(
                        s * preprocess_config["preprocessing"]["audio"]["sampling_rate"] /
                        preprocess_config["preprocessing"]["stft"]["hop_length"])
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

    # if not (word2phones[0][1][0][-1] == "" and word2phones[-1][1][0][-1] == ""):
    #     import pdb
    #     pdb.set_trace()
    return phones, durations, start_time, end_time


def output_phones(out, input_file, output_file):
    with open(input_file) as fp_in, open(output_file, 'w') as fp_out:
        for line in fp_in:
            name = line.strip().split("|")[0]
            if name not in out:
                print(f"Skip {name} since it is not found in the TextGrid")

            if len(out[name]) == 2:
                # without sid
                fp_out.write('{}|{}\n'.format(out[name][0], ' '.join(out[name][1])))
            else:
                # with sid
                fp_out.write('{}|{}|{}\n'.format(out[name][0], out[name][2], ' '.join(out[name][1])))
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--words", action="store_true", help="add #0 between words")
    parser.add_argument("--train_txt_fpath", type=str, default="train.txt")
    parser.add_argument("--val_txt_fpath", type=str, default=None)
    parser.add_argument("--test_txt_fpath", type=str, default=None)
    parser.add_argument("--use_sid", action="store_true", help="add speaker id to the labels")
    parser.add_argument(
        "--remove_silence",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="To remove the heading and tailing silence"
    )
    parser.add_argument("-p", "--preprocess_config", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    raw_path = preprocess_config["path"]["raw_path"]
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]

    spk_dirs = [
        speaker for speaker in os.listdir(os.path.join(preprocessed_path, "TextGrid"))
        if os.path.isdir(os.path.join(preprocessed_path, "TextGrid", speaker))
    ]
    # Sort the speakers by the names
    spk_dirs = sorted(spk_dirs)
    print("all spks {}".format(spk_dirs))
    print("Total num of spks: {}".format(len(spk_dirs)))

    if args.use_sid:
        spk2id = {}
        for sid, speaker in enumerate(spk_dirs):
            spk2id[speaker] = sid

    print(spk2id)
    sos_sil_stat, eos_sil_stat, sil_stat = stats_sil_phones_from_path(preprocess_config["path"]["preprocessed_path"])
    spk2pause_stats = dict()
    for spk in sos_sil_stat.keys():
        print("====={}".format(spk))
        pos_stats = pd.Series(sil_stat[spk]).describe(percentiles=[.3, .6, .8, .99])
        spk2pause_stats[spk] = [pos_stats["30%"], pos_stats["60%"], pos_stats["80%"]]

    out = {}
    for speaker in spk_dirs:
        for tg_name in os.listdir(os.path.join(preprocessed_path, "TextGrid", speaker)):
            if ".TextGrid" not in tg_name:
                continue
            basename = tg_name.split(".")[0]
            wav_path = os.path.join(raw_path, speaker, "{}.wav".format(basename))

            phones, duration, start, end = get_phones(
                args.words,
                preprocess_config,
                spk2pause_stats[speaker],
                os.path.join(preprocessed_path, "TextGrid", speaker, tg_name)
            )

            sr, wav = wavfile.read(wav_path)
            assert sr == preprocess_config["preprocessing"]["audio"]["sampling_rate"]

            if args.remove_silence:
                # trim wav
                wav = wav[
                    max(int(preprocess_config["preprocessing"]["audio"]["sampling_rate"] * start), 0) :
                        int(preprocess_config["preprocessing"]["audio"]["sampling_rate"] * end)
                ]
            else:
                # add silence to the label
                phones = ['sil'] + phones + ['sil']
                
            # write wave
            os.makedirs(os.path.join(args.output_dir, "Wave", speaker), exist_ok=True)
            out_wav_path = os.path.join(args.output_dir, "Wave", speaker, "{}.wav".format(basename))
            wavfile.write(out_wav_path, sr, wav.astype(np.int16))
            if args.use_sid:
                out[basename] = [out_wav_path, phones, spk2id[speaker]]
            else:
                out[basename] = [out_wav_path, phones]
    
    output_phones(
        out,
        os.path.join(preprocessed_path, args.train_txt_fpath),
        os.path.join(args.output_dir, args.train_txt_fpath)
    )
    if args.val_txt_fpath is not None:
        output_phones(
            out,
            os.path.join(preprocessed_path, args.val_txt_fpath),
            os.path.join(args.output_dir, args.val_txt_fpath)
        )
    if args.test_txt_fpath is not None:
        output_phones(
            out,
            os.path.join(preprocessed_path, args.test_txt_fpath),
            os.path.join(args.output_dir, args.test_txt_fpath)
        )

    if args.use_sid:
        with open(os.path.join(args.output_dir, "speakers.json"), 'w') as f:
            json.dump(spk2id, f)


if __name__ == '__main__':
    main()