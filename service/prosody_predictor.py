import copy
import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, BertConfig, BertLayer
from transformers import AutoTokenizer

try:
    import onnxruntime as ort
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)

from . import rhy_postprocess

TARGET_DEVICE = os.getenv("TARGET_DEVICE", "")
if TARGET_DEVICE == "mtgpu":
    import musa_torch_extension
    MTGPU_DEVICE = torch.device(TARGET_DEVICE)
elif TARGET_DEVICE == "":
    TARGET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FrontendModel(nn.Module):

    def __init__(self, num_phones: int, num_prosody: int, model_path: str):
        super(FrontendModel, self).__init__()
        config = BertConfig.from_pretrained(
            os.path.join(model_path, "config.json"))
        self.bert = AutoModel.from_pretrained(os.path.join(
            model_path, "model.pt"),
                                              config=config)
        for param in self.bert.parameters():
            param.requires_grad_(False)
        d_model = self.bert.config.to_dict()['hidden_size']
        assert d_model in (
            768, 1024), 'Expected bert encoder input dim is 768 or 1024'
        self.transform = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=8,
                                                    dim_feedforward=2048,
                                                    batch_first=True)
        self.phone_classifier = nn.Linear(d_model, num_phones)
        self.prosody_classifier = nn.Linear(d_model, num_prosody)

    def _forward(self, x):
        # mask = x['attention_mask'] == 0
        mask = None
        bert_output = self.bert(**x)
        x = self.transform(bert_output.last_hidden_state,
                           src_key_padding_mask=mask)
        phone_pred = self.phone_classifier(x)
        prosody_pred = self.prosody_classifier(x)
        return phone_pred, prosody_pred

    def forward(self, x):
        return self._forward(x)

    def export_forward(self, x):
        assert x.size(0) == 1
        x = {
            'input_ids':
            x,
            'token_type_ids':
            torch.zeros(1, x.size(1), dtype=torch.int64).to(x.device),
            'attention_mask':
            torch.ones(1, x.size(1), dtype=torch.int64).to(x.device)
        }
        phone_logits, prosody_logits = self._forward(x)
        phone_pred = F.softmax(phone_logits, dim=-1)
        prosody_pred = F.softmax(prosody_logits, dim=-1)
        return phone_pred, prosody_pred


def _add_prosody_into_text(text, prosody):
    lst = []
    for ph, rhy in zip(text, prosody):
        if ph != 'UNK':
            lst.append(ph)
        if rhy == 4:
            continue
        if rhy != 0:
            # lst.append('sp' + str(rhy))
            lst.append(f' #{rhy} ')
    return lst


def _remove_prosody(text, use_sp=False):
    if use_sp:
        pattern = r' (sp[0-4]) '
    else:
        pattern = r' (#[0-4]) '
    lst = re.sub(pattern, r'_\1_', text).split('_')

    rank_lst = []
    char_lst = []
    preffix_sum = 0
    for x in lst:
        if re.match('(sp|#)[0-4]', x):
            rank_lst.append((preffix_sum, x))
        else:
            char_lst.append(x)
            preffix_sum += len(x)
    return rank_lst, ''.join(char_lst)


def _combine_prosody(text, rank_lst, pred_rank_lst, min_dist=4):
    d = dict(pred_rank_lst)
    logging.info(f'pred prosody dict: {d}')
    logging.info(f'artifact prosody list: {rank_lst}')
    for pos, _ in rank_lst:
        for i in range(max(0, pos - min_dist), pos + min_dist):
            if i in d:
                logging.info(
                    f'ignore prosody nearby artifact pos {pos}: {i} {d[i]}')
                del d[i]
    for pos, rank in rank_lst:
        if rank in d:
            logging.info(f'use artifact over pred: {rank} => {d[pos]}')
        else:
            logging.info(f'add artifact rank: pos={pos} rank={rank}')
        d[pos] = rank

    char_lst = list(text)
    for pos in sorted(d, reverse=True):
        char_lst.insert(pos, f' {d[pos]} ')
        # logging.debug(f'char_lst: {char_lst}')
    return ''.join(char_lst)


class FrontendPtRuntime(object):

    def __init__(
            self,
            model_dir: str,
            device: str = "cpu",
            num_phones: int = 876,  # local/polyphone_phone.txt
            num_prosody: int = 5,  # local/prosody2id.txt
    ):
        model = FrontendModel(num_phones, num_prosody, model_dir)

        model_path = os.path.join(model_dir, 'model.pt') if os.path.exists(os.path.join(model_dir, 'model.pt')) \
            else os.path.join(model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)

        self.device = device
        if TARGET_DEVICE == "mtgpu":
            model.to(MTGPU_DEVICE)
            self.device = MTGPU_DEVICE

        model.eval()

        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer"))

    def run(self, x):
        with torch.inference_mode(mode=True):
            # tokens = self.tokenizer.encode(x, add_special_tokens=False)
            inputs = self.tokenizer(list(x),
                                    padding=True,
                                    truncation=True,
                                    is_split_into_words=True,
                                    return_tensors="pt")['input_ids']
            inputs = inputs.to(self.device)
            phone_pred, prosody_pred = self.model.export_forward(inputs)
        return phone_pred, prosody_pred


class FrontendOnnxRuntime(object):

    def __init__(self, model_dir: str, device="cpu"):
        provider_map = {
            'cpu': 'CPUExecutionProvider',
            'cuda': 'CUDAExecutionProvider',
            'tensorrt': 'TensorrtExecutionProvider',
            # 'mtgpu': 'xxx',
        }
        self.ppm_sess = ort.InferenceSession(os.path.join(
            model_dir, "model.onnx"),
                                             providers=[
                                                 provider_map[device],
                                             ])
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer"))

    def run(self, x):
        tokens = self.tokenizer(list(x),
                                is_split_into_words=True,
                                return_tensors="np")['input_ids']
        ort_inputs = {'input': tokens}
        phone_pred, prosody_pred = self.ppm_sess.run(None, ort_inputs)
        return phone_pred, prosody_pred


class ProsodyPredictor:

    def __init__(self, model_dir, device="cpu"):
        if os.path.exists(os.path.join(
                model_dir, "model.pt")) or os.path.exists(
                    os.path.join(model_dir, "model.pth")):
            runtime_cls = FrontendPtRuntime
        elif os.path.exists(os.path.join(model_dir, "model.onnx")):
            runtime_cls = FrontendOnnxRuntime
        else:
            raise NotImplementedError(
                "there is no model.pt|model.onnx|model.pth")
        assert os.path.isdir(os.path.join(model_dir, "tokenizer"))

        logging.info(f'use runtime {runtime_cls}')

        self.sess = runtime_cls(model_dir, device=device)

        return

    def g2p(self, x):
        pinyin = None
        _, prosody_predict = self.sess.run(x)
        # NOTE(yichao): onnx model can only return ndarray, while pt model return torch tensor
        if not isinstance(prosody_predict, np.ndarray):
            prosody_predict = prosody_predict.cpu().numpy()
        prosody_pred = prosody_predict.argmax(-1)[0][1:-1]
        return pinyin, prosody_pred, x

    def _predict_zh_prosody(self, text, use_pinyin=False):
        pinyin, prosody, hanzi = self.g2p(text)
        logging.debug('pinyin: %s', pinyin)
        logging.debug('prosody: %s', prosody)
        logging.debug('hanzi: %s', hanzi)

        if use_pinyin:
            symbols = pinyin
        else:
            # symbols = re.sub(r'\W', '', hanzi)
            symbols = hanzi

        return _add_prosody_into_text(symbols, prosody)

    def _predict_prosody(
            self,
            text,
            use_pinyin=False,
            min_text_len=rhy_postprocess.PROSODY_PAUSE_LENGTH_THRESHOLD):
        """
        Args:
            @min_text_len: do not run prosody inference, if text len less than min_text_len
        """

        def _split_sentence(text):
            segment_lst = re.sub(rf'([{sentence_sep}]+)', r'_\1_',
                                 text).split('_')
            return segment_lst

        sentence_sep = r'\Wa-zA-Z\s'  # 区分中英文
        segment_lst = _split_sentence(text)
        if len(segment_lst) > 1:
            logging.info('split sentence segment list: %s', segment_lst)

        lst = []
        for seg in segment_lst:
            if not seg:
                continue
            if not re.search(rf'([{sentence_sep}]+)', seg):
                if len(seg) > min_text_len:
                    seg_lst = self._predict_zh_prosody(seg,
                                                       use_pinyin=use_pinyin)
                    lst += seg_lst
                else:
                    # logging.warning(f'skip prosody prediction, text_len {len(seg)} < min_text_len {min_text_len}')
                    lst.append(seg)
            else:
                lst.append(seg)

        if use_pinyin:
            pinyin_str = ' '.join(lst)
        else:
            pinyin_str = ''.join(lst)
        return pinyin_str

    def predict(self,
                text: str,
                use_sp: bool = False,
                use_postprocess=True,
                use_artifact=False) -> str:
        """
        摩尔线程真棒 => 摩尔线程 sp0 真棒

        Args:
            @use_sp: with `sp0/sp1` if `use_sp=True`; otherwise `#2/#3` by default.
            @use_postprocess: apply postprocess rules to adjust prosody.
            @use_artifact: parse prosody `sp0/sp1` or `#2/#3` in text added by human.
        """
        # text is hanzi
        if use_artifact:
            rank_lst, text = _remove_prosody(text, use_sp=use_sp)

        # pinyin, prosody, hanzi = self.frontend.g2p(text)
        # seg_lst = _add_prosody_into_text(text, prosody)
        # text = ''.join(seg_lst)

        text = self._predict_prosody(text)

        if use_postprocess:
            text = rhy_postprocess.process_text(text)

        if use_sp:
            text = text.replace('#2', 'sp0').replace('#3', 'sp1')

        if use_artifact and rank_lst:
            pred_rank_lst, text = _remove_prosody(text, use_sp=use_sp)
            text = _combine_prosody(text, rank_lst, pred_rank_lst)

        return text

