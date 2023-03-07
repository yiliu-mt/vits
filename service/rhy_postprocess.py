"""
Function: 在预测 Prosody 结果后，加入策略，调整停顿标记。
"""

import logging
import re
# logging.basicConfig(level='DEBUG')


PROSODY_PAUSE_LENGTH_THRESHOLD = 12
PROSODY_MIN_DIST = 6
PROSODY_RANK_1 = ' #1 '
PROSODY_RANK_2 = ' #2 '
PROSODY_RANK_3 = ' #3 '


def get_xml_break(sp):
    # return f'<break time="{sp}"/>'
    return f' {sp} '


def ignore_too_short_rank(text, rank, min_dist=PROSODY_MIN_DIST):
    lst = text.split(rank)
    lst = [x.strip() for x in lst]
    if len(lst) > 1:
        if len(lst[0]) < min_dist:
            logging.info(f'ignore {rank} at the begin of {text}')
            lst[1] = lst[0] + lst[1]
            lst.pop(0)
    if len(lst) > 1:  # 2段以上，就区分头尾
        if len(lst[-1]) < min_dist:
            logging.info(f'ignore {rank} at the end of {text}')
            lst[-2] = lst[-2] + lst[-1]
            lst.pop(-1)
    logging.debug(f'too short lst: {lst}')
    return rank.join(lst)


def compress_too_close_rank(text, rank, min_dist=PROSODY_MIN_DIST):
    lst = text.split(rank)
    lst = [x.strip() for x in lst]
    rank_lst = []
    for i, x in enumerate(lst):
        if len(x) < min_dist:
            rank_lst.append('')
        else:
            rank_lst.append(rank)
    res = []
    for x, y in zip(lst, rank_lst):
        res.append(x)
        res.append(y)
    res = res[:-1]  # drop the last rank
    return ''.join(res)


def adjust_prosody_v1(text, len_thresh=16):
    """
    Strategy:
        * #3 分割子句
        * 对 #2 进行处理
    """
    if not text:
        return text

    rank3 = PROSODY_RANK_3
    rank2 = PROSODY_RANK_2

    lst = text.split(rank3)  # 以 rank3 为边界，分割子句

    res_lst = []
    for sub_text in lst:
        # avoid "理解算法 #2 等 #3 都是"
        sub_text = ignore_too_short_rank(sub_text, rank2, min_dist=PROSODY_MIN_DIST)
        # too short sub sentence remove rank2
        if len(sub_text) < len_thresh or rank2 not in sub_text:
            sub_text = sub_text.replace(rank2, '')  # ignore

        res_lst.append(sub_text)
    return rank3.join(res_lst)


def remove_rank(text, rank):
    return text.replace(rank, '')


def process_rhy(text):
    """
    Method:
        1. 如果有 #3， 对应 sp1；
        2. 如果没有 #3，启用 #2，对应 sp0；
            * 一定距离 M
                * 经 case by case 分析，暂定 M 是 PROSODY_MIN_DIST =6
                * 多个 #2，将距离小于 M 的合并为一个
                * 多个 #3，将距离小于 M 的合并为一个
            * 一定长度 N
                * 经统计分析，暂定 N 是 PROSODY_PAUSE_LENGTH_THRESHOLD=12
                * N 字以上，需要韵律预测
                * N 字以下，不启用韵律
    """
    # 删除 #1
    text = remove_rank(text, PROSODY_RANK_1)

    # 忽略首尾小于 min_dist 的 prosody 标记
    text = ignore_too_short_rank(text, PROSODY_RANK_3)
    text = ignore_too_short_rank(text, PROSODY_RANK_2)

    # 合并距离小于 min_dist 的 prosody 标记
    text = compress_too_close_rank(text, PROSODY_RANK_3)
    text = compress_too_close_rank(text, PROSODY_RANK_2)

    # 对 #3 分割的子句中的 #2 进行处理
    text = adjust_prosody_v1(text)
    return text


def process_text(text):
    lst = [process_rhy(x) for x in split_sentences(text)]
    return ''.join(lst)


def split_sentences(line):
    sentence_sep = '-,.!?;:：；，。！？、《》——'
    sub_sentence_lst = re.sub(rf'([{sentence_sep}])', r'\1\n',
                              line.strip()).split('\n')
    if '' in sub_sentence_lst:
        sub_sentence_lst.remove('')
    logging.debug(f'sub_sentence_lst: {sub_sentence_lst}')
    return sub_sentence_lst