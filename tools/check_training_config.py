import argparse
import os
import json


def read_train_list(filename):
    speaker_map = {}
    with open(filename) as f:
        for line in f:
            path, id, _ = line.strip().split("|")
            speaker_name = os.path.dirname(path).rsplit("/", 1)[-1]
            assert int(id) >= 0
            if speaker_name not in speaker_map:
                speaker_map[speaker_name] = int(id)
            else:
                assert speaker_map[speaker_name] == int(id)
    return speaker_map


def check_val_list(filename, speaker_map):
    with open(filename) as f:
        for line in f:
            path, id, _ = line.strip().split("|")
            speaker_name = os.path.dirname(path).rsplit("/", 1)[-1]
            assert speaker_map[speaker_name] == int(id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    print("num of speakers: {}".format(config["data"]["n_speakers"]))

    speaker_map = read_train_list(config["data"]["training_files"])
    check_val_list(config["data"]["validation_files"], speaker_map)
    max_speaker_id = max([item[1] for item in speaker_map.items()])
    print("Max speaker id: {}".format(max_speaker_id))
    print("Check passed.")


if __name__ == '__main__':
    main()

