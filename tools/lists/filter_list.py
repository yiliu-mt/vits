import argparse
import json
import os


def read_speaker_map(speaker_file):
    return json.load(open(speaker_file))

    
def read_list(input_list):
    filelist = []
    with open(input_list) as f:
        for line in f:
            path, id, label = line.split("|")
            filelist.append((path, int(id), label))
    return filelist

def process(filelist, speaker_map, include, exclude, use_new_speaker_map, output_speakers):
    inverse_speaker_map = {}
    for item in speaker_map.items():
        inverse_speaker_map[item[1]] = item[0]

    is_include = True if include is not None else False
    if is_include:
        include = [speaker_map[i] for i in include.split('|')]
    else:
        exclude = [speaker_map[i] for i in exclude.split('|')]

    if use_new_speaker_map:
        new_speaker_map = read_speaker_map(output_speakers)
    else:
        new_speaker_map = {}

    new_filelist = []
    for item in filelist:
        if is_include:
            if item[1] not in include:
                continue
        else:
            if item[1] in exclude:
                continue
        
        if not use_new_speaker_map:
            if inverse_speaker_map[item[1]] not in new_speaker_map:
                new_speaker_map[inverse_speaker_map[item[1]]] = len(new_speaker_map.keys())
        new_id = new_speaker_map[inverse_speaker_map[item[1]]]
        new_filelist.append((item[0], str(new_id), item[2]))

    return new_filelist, new_speaker_map


def output_speaker_map(speaker_map, output_speakers):
    json.dump(speaker_map, open(output_speakers, 'w'))


def output_list(filelist, output_list):
    with open(output_list, 'w') as f:
        for item in filelist:
            f.write('|'.join(item))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", type=str, default=None, help="speaker names to include, delimited by \"|\"")
    parser.add_argument("--exclude", type=str, default=None, help="speaker names to exclude, delimited by \"|\"")
    parser.add_argument("--use-new-speaker-map", action="store_true", help="If true, read the output_speaker_map and output id using that file.")
    parser.add_argument("input_speakers")
    parser.add_argument("input_list")
    parser.add_argument("output_speakers")
    parser.add_argument("output_list")
    args = parser.parse_args()

    assert args.include is not None or args.exclude is not None, "You should specify --include or --exclude"
    if args.use_new_speaker_map:
        assert os.path.exists(args.output_speakers)
        
    speaker_map = read_speaker_map(args.input_speakers)
    filelist = read_list(args.input_list)
    filelist, speaker_map = process(
        filelist, speaker_map, args.include, args.exclude, use_new_speaker_map=args.use_new_speaker_map, output_speakers=args.output_speakers)
    
    if not args.use_new_speaker_map:
        output_speaker_map(speaker_map, args.output_speakers)
    output_list(filelist, args.output_list)


if __name__ == "__main__":
    main()
