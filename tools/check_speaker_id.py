import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("speaker_json")
    parser.add_argument("text")
    args = parser.parse_args()

    spk2id = json.load(open(args.speaker_json))
    with open(args.text) as f:
        for line in f:
            path, id, _ = line.strip().split("|")
            speaker_name = os.path.dirname(path).split("/")[-1]
            if spk2id[speaker_name] != int(id):
                print(f"Error: speaker {speaker_name} != id {id}")
                quit()

if __name__ == '__main__':
    main()
