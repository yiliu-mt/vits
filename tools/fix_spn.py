import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    utt2phone = {}
    with open(args.reference) as f:
        for line in f:
            utt, _, phone = line.split("|")[:3]
            utt2phone[utt] = phone[1:-1].split(" ")
    
    with open(args.input) as fp_in, open(args.output, 'w') as fp_out:
        for line in fp_in:
            wavpath, spk, phone, dur = line.strip().split("|")
            utt = wavpath.rsplit("/", 1)[1].rsplit(".", 1)[0]
            if utt not in utt2phone:
                continue
            phone = phone.split(" ")
            new_phone = []
            ref_idx, phone_idx = 0, 0
            while phone_idx < len(phone):
                if phone[phone_idx][0] == '#':
                    new_phone.append(phone[phone_idx])
                    phone_idx += 1
                    continue
                elif phone[phone_idx].startswith("sp"):
                    assert utt2phone[utt][ref_idx].startswith("sp")
                else:
                    assert phone[phone_idx] == utt2phone[utt][ref_idx]
                new_phone.append(utt2phone[utt][ref_idx])
                ref_idx += 1
                phone_idx += 1

            assert ref_idx == len(utt2phone[utt]) and phone_idx == len(phone)
            fp_out.write("{}|{}|{}|{}\n".format(wavpath, spk, ' '.join(new_phone), dur))
    

if __name__ == '__main__':
    main()
