import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    utt2prosody = {}
    with open(args.reference, 'r') as f:
        for line in f:
            utt, _, phone, _, prosody = line.strip().split("|")
            phone = phone[1:-1].split(" ")
            utt2prosody[utt] = (phone, prosody)
    
    with open(args.input) as fp_in, open(args.output, 'w') as fp_out:
        for line in fp_in:
            wavpath, spk, phone, dur = line.strip().split("|")
            utt = wavpath.rsplit("/", 1)[1].split(".")[0]
            if utt not in utt2prosody:
                continue
            phone = phone.split(" ")
            phone_idx, ref_idx = 0, 0
            while phone_idx < len(phone):
                if phone[phone_idx].startswith("sp"):
                    phone_idx += 1
                elif utt2prosody[utt][0][ref_idx].startswith("sp"):
                    ref_idx += 1
                else:
                    if phone[phone_idx] != utt2prosody[utt][0][ref_idx]:
                        import pdb
                        pdb.set_trace()
                    phone_idx += 1
                    ref_idx += 1

            fp_out.write("{}|{}|{}|{}|{}\n".format(
                wavpath, spk, ' '.join(utt2prosody[utt][0]), dur, utt2prosody[utt][1]
            ))


if __name__ == '__main__':
    main()