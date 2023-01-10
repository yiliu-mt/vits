# make training list from the FastSpeech List
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', type=str, required=True)
    parser.add_argument('-i', '--input_list', type=str, required=True)
    parser.add_argument('-o', '--output_list', type=str, required=True)
    args = parser.parse_args()
    
    utt2info = {}
    with open(args.input_list) as f:
        for line in f:
            path = line.split('|')[0]
            uttid = os.path.basename(path).rsplit('.', 1)[0]
            utt2info[uttid] = line

    with open(args.reference) as fp_in, open(args.output_list, 'w') as fp_out:
        for line in fp_in:
            uttid = line.split('|')[0]
            if uttid not in utt2info:
                print(f"Cannot find {uttid} from the input list.")
                continue
            fp_out.write(utt2info[uttid])
        

if __name__ == '__main__':
    main()