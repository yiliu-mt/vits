import sys
import random

input_text = sys.argv[1]
input_ssml = sys.argv[2]
output_text = sys.argv[3]
output_ssml = sys.argv[4]

utt2text = {}
with open(input_text, 'r') as f:
    for line in f:
        utt, text = line.strip().split(" ", 1)
        utt2text[utt] = text

utt2ssml = {}
with open(input_ssml, 'r') as f:
    for line in f:
        utt, ssml = line.strip().split(" ", 1)
        utt2ssml[utt] = ssml

random.seed(42)
utt_list = list(utt2text.keys())
random.shuffle(utt_list)

with open(output_text, 'w') as fp_text, open(output_ssml, 'w') as fp_ssml:
    for i, utt in enumerate(utt_list):
        fp_text.write('{} {}\n'.format(i, utt2text[utt]))
        fp_ssml.write('{} {}\n'.format(i, utt2ssml[utt]))
