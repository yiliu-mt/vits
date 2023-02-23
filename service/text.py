import re
import logging
from string import punctuation
import numpy as np
from g2p_en import G2p
from pypinyin import pinyin, Style
from text import text_to_sequence


def soft_cut(para):
    para = para.replace('|||', "\n")  # 智能客服faq中的分段逻辑
    para = re.sub('([。！？；\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？；\?][”’])([^，。！？；\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return [utt for utt in para.split("\n") if utt.strip()]


def soft_cut_by_comma(para):
    para = re.sub('([，,：])([^”’])', r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocess_req_text(text, max_length=50, cut_func=soft_cut):
    soft_cut_sents = []
    cur_sent = ""
    for sent in cut_func(text):
        sent = clean_text(sent)
        sent = sent.strip()
        cur_sent = cur_sent.strip()
        if cur_sent and len(cur_sent) + len(sent) > max_length:
            soft_cut_sents.append(cur_sent.strip())
            cur_sent = ""
        cur_sent += sent
    if cur_sent:
        soft_cut_sents.append(cur_sent.strip())
    return soft_cut_sents

 
def preprocess_text_to_sequence(config, phone_seq, syllable_seq):
    phones_str = "{" + " ".join(phone_seq) + "}"
    logging.info("Phoneme Sequence: {}".format(phones_str))
    sequence = np.array(
        text_to_sequence(
            phones_str, [],
            symbol_version=config.data.symbol_version
        )
    )
    assert len(phone_seq) == len(syllable_seq) == len(sequence), "{} != {} != {}".format(len(phone_seq),
                                                                                         len(syllable_seq),
                                                                                         len(sequence))
    return sequence, syllable_seq, phone_seq


def convert_punc_to_sp(text):
    EOS_PAUSE_PUNCS_ZH = '！？｡。'
    EOS_PAUSE_PUNCS_EN = '!\?'
    LONG_PAUSE_PUNCS_ZH = '，：；,;:'
    LONG_PAUSE_PUNCS_EN = ',;:'
    SHORT_PAUSE_PUNCS_ZH = '＂＃＄％＆＇（）＊＋－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏·〈〉\-'
    SHORT_PAUSE_PUNCS_EN = '\\`\{|\}#\$%&\(\)\*\+\-\/<=>@\[\]\^_~'
    SPECIAL_PUNCS_EN = """'."""
    text = re.sub('[{}{}]'.format(EOS_PAUSE_PUNCS_ZH, EOS_PAUSE_PUNCS_EN), " sp3 ", text)
    text = re.sub('[{}{}]'.format(LONG_PAUSE_PUNCS_ZH, LONG_PAUSE_PUNCS_EN), " sp2 ", text)
    text = re.sub('[{}{}]'.format(SHORT_PAUSE_PUNCS_ZH, SHORT_PAUSE_PUNCS_EN), " sp1 ", text)
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocess_text_to_syllable(text, lexicon):
    def handle_unmatch_symbols(_text):
        # replace punc to sp or remove
        _text = convert_punc_to_sp(_text).strip()
        words = re.split(r"([\s+])", _text)
        syllables = []
        for w in words:
            if len(w.strip()) == 0:
                continue
            is_contain_alpha = False
            for c in w:
                if c.isalnum():
                    is_contain_alpha = True
                    break
            if not is_contain_alpha:
                logging.warn("ILLEGAL WORDS [{}]".format(w))
                continue
            if w in ["sp1", "sp2", "sp3", "sp4", "sp0", "sp"]:
                syllables.append(w)
            elif w in lexicon:
                syllables.append(w)
            elif w.isupper() and len(w) <= 3:
                for c in w:
                    syllables.append(c)
            elif w.lower() in lexicon:
                syllables.append(w.lower())
            elif w.isupper():
                for c in w:
                    syllables.append(c)
            else:
                logging.warn("OOV in word found [{}] in text [{}]".format(w, text))
                syllables.append(w)
        return syllables

    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True, heteronym=False,
            errors=handle_unmatch_symbols,
        )
    ]
    logging.info("Syllable Sequence: {}".format(pinyins))
    return pinyins


def preprocess_syllable_to_phoneme(pinyins, lexicon, oov_lexicon, default_sp="sp2"):
    syllable_seq = []
    phone_seq = []
    for i, p in enumerate(pinyins):
        if p in lexicon:
            phone_seq += lexicon[p]
            syllable_seq += [(len(syllable_seq), p)] * len(lexicon[p])
        elif p in ["sp1", "sp2", "sp3", "sp4", "sp0", "sp"]:
            phone_seq.append(p)
            syllable_seq.append((len(syllable_seq), p))
        else:
            p = p.lower().strip()
            if p in oov_lexicon:
                phone_seq += oov_lexicon[p]
                syllable_seq += [(len(syllable_seq), p)] * len(oov_lexicon[p])
            else:
                for p_char in p:
                    if p_char in oov_lexicon:
                        phone_seq += oov_lexicon[p_char]
                        syllable_seq += [(len(syllable_seq), p_char)] * len(oov_lexicon[p_char])
                    else:
                        if i < len(pinyins) - 1:
                            phone_seq.append(default_sp)
                            syllable_seq.append((len(syllable_seq), default_sp))
    return phone_seq, syllable_seq


def preprocess_text(config, text, lexicon, oov_lexicon):

    pinyins = preprocess_text_to_syllable(text, lexicon)

    if pinyins and pinyins[-1] in ["sp1", "sp2", "sp3", "sp4", "sp0", "sp"]:
        pinyins.pop()
    for i in range(len(pinyins)):
        if pinyins[i] in ["sp1", "sp2", "sp3", "sp4", "sp0", "sp"] and config.data.symbol_version != "v2.1":
            pinyins[i] = "sp"

    default_sp = "sp" if config.data.symbol_version != "v2.1" else "sp2"
    phone_seq, syllable_seq = preprocess_syllable_to_phoneme(pinyins, lexicon, oov_lexicon, default_sp=default_sp)
    return preprocess_text_to_sequence(config, phone_seq, syllable_seq)

