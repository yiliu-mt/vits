import re
from collections import defaultdict


def read_lexicon(lex_path, lower=True):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if lower:
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
            else:
                if word not in lexicon:
                    lexicon[word] = phones
    return lexicon


def read_oov_lexicon(lexicon, oov_mapping_path):
    oov_mapping_dict = dict()
    for line in open(oov_mapping_path):
        oov, oov_syls = line.strip().split(" ", 1)
        oov_syls = oov_syls.split(" ")
        oov_phones = []
        for oov_syl in oov_syls:
            assert oov_syl in lexicon, "{} is not in lexicon".format(oov_syl)
            oov_phones.extend(lexicon[oov_syl])
        oov_mapping_dict[oov] = oov_phones
    return oov_mapping_dict


def read_fast_map_dict(fpath):
    fast_match_dict = defaultdict(dict)
    for line in open(fpath):
        utt_id, spk_id, phone, _, text = line.strip().split("|")
        phones_seq = []
        syllables_seq = []
        sylable_phones = phone.split(" - ")
        for syllable_phone in sylable_phones:
            syl_phones = syllable_phone.strip().split(" ")
            for phone in syl_phones:
                syllables_seq.append((len(phones_seq), "xxx"))
            for phone in syl_phones:
                phones_seq.append(phone)
        fast_match_dict[spk_id][text] = (syllables_seq, phones_seq)
    return fast_match_dict