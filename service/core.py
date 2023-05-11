import re
import time
import json
import requests
import copy
import logging
import traceback
import utils
from collections import defaultdict
from typing import List
import torch
import numpy as np
import commons
from models import SynthesizerTrn
from text.symbols import get_symbols
from utils import waveform_postprocessing
from .ssml import parse_ssml
from .lexicon import read_lexicon, read_oov_lexicon, read_fast_map_dict
from .cn_tn import TextNorm
from .prosody_predictor import ProsodyPredictor
from .text import preprocess_req_text, soft_cut_by_comma, \
    preprocess_text_to_sequence, preprocess_text


class Utterance:
    def __init__(self, utt_id, text, voice):
        self.utt_id = utt_id
        self.text = text
        self.voice = voice

        self.sample_rate = None
        self.normed_text = None
        self.normed_text_with_prosody = None
        self.syllable_seq = None
        self.phone_seq = None
        self.phone_id_seq = None

        self.pred_wav = None
        self.sos_sil = 0.05
        self.eos_sil = 0.25
        self.phone_speed_seq = None
        self.phone_pitch_seq = None
        self.phone_energy_seq = None

        self.noise_scale = 0.667
        self.noise_scale_w = 0.8

    @staticmethod
    def merge_multi_utterances_audio(utterances, recording_resource_map=None):
        assert len(utterances) > 0
        audio_data_list = []
        sample_rate = utterances[0].sample_rate
        for utt in utterances:
            assert isinstance(utt, Utterance)
            assert sample_rate == utt.sample_rate
            audio_data_list.append(utt.get_padded_audio_data(recording_resource_map))
        audio_data = np.concatenate(audio_data_list, axis=-1).astype(np.int16)
        return audio_data
    
    def get_padded_audio_data(self, recording_resource_dict=None):
        assert self.pred_wav is not None
        pad_start_sil = self.sos_sil
        pad_end_sil = self.eos_sil

        if recording_resource_dict is None or "SIL" not in recording_resource_dict:
            audio_data = np.pad(self.pred_wav,
                                pad_width=[int(pad_start_sil * self.sample_rate), int(pad_end_sil * self.sample_rate)],
                                mode='constant')
        else:
            padded_data = []
            sil_audio_data = recording_resource_dict["SIL"]

            # TODO (yi.liu): the postion of the noise
            def _slice_or_slice_after_tile_recording_audio(data, n):
                if len(data) >= n:
                    return data[:n]
                else:
                    return np.tile(data, n//len(data) + 1)[:n]
            padded_data.append(_slice_or_slice_after_tile_recording_audio(sil_audio_data, int(pad_start_sil * self.sample_rate)))
            padded_data.append(self.pred_wav)
            padded_data.append(_slice_or_slice_after_tile_recording_audio(sil_audio_data, int(pad_end_sil * self.sample_rate)))
            audio_data = np.concatenate(padded_data)
        return audio_data


class GenerationRequest:

    def __init__(self, text=None, ssml=None, voice=None, noise_scale=0.667, noise_scale_w=0.8, speed_rate=1.0, version=None, format="pcm", **_kwargs):
        self.text = text
        if (not ssml) and text.strip() and text.strip().startswith("<speak>") and text.strip().endswith("</speak>"):
            self.ssml = text.strip()
        else:
            self.ssml = ssml
        self.speed_rate = speed_rate
        self.voice = voice
        self.version = version
        self.format = format
        self.noise_scale = noise_scale
        self.noise_scale_w = noise_scale_w

    def __str__(self):
        return str(self.__dict__)


class TTS:
    '''VITS service class
    '''
    def __init__(self, config_file, model_file, lexicon, device, fast_match_fpath=None, prosody_model=None):
        # Load config and model
        self.__hps = utils.get_hparams_from_file(config_file)
        self.__model = SynthesizerTrn(
            len(get_symbols(self.__hps.data.get("symbol_version", "default"))),
            self.__hps.data.filter_length // 2 + 1,
            self.__hps.train.segment_size // self.__hps.data.hop_length,
            n_speakers=self.__hps.data.n_speakers,
            **self.__hps.model).to(device)
        self.__model.eval()
        self.__device = device
        utils.load_checkpoint(model_file, self.__model, None)
        logging.info(f"Loaded model from {model_file}")

        # Load lexicon
        self.lexicon = read_lexicon(lexicon, lower=False)
        self.oov_lexicon = read_oov_lexicon(self.lexicon, "service/oov-mapping.txt")
        self.fast_match_dict = defaultdict(dict)
        if fast_match_fpath is not None:
            self.fast_match_dict = read_fast_map_dict(fast_match_fpath)
        self.text_normalizer = TextNorm(remove_punc=False)
        self.prosody_predictor = None
        if prosody_model is not None:
            self.prosody_predictor = ProsodyPredictor(prosody_model)

    @property
    def sample_rate(self):
        return self.__hps.data.sampling_rate

    @property
    def config(self):
        return self.__hps
    
    def init_utterance(self, utt_id, text, voice):
        return Utterance(utt_id, text, voice)
    
    def init_phone_utterance(self, utt_id, text, voice):
        utterance = Utterance(utt_id, text, voice)
        # Make a dummy syllable_seq
        syllable_seq = phone_seq = text.strip().split(" ")
        phone_id_seq, syllable_seq, phone_seq = preprocess_text_to_sequence(
            self.__hps, phone_seq, syllable_seq)
        utterance.phone_id_seq = phone_id_seq
        utterance.phone_seq = phone_seq
        utterance.syllable_seq = syllable_seq
        return utterance
    
    def preprocess(self, utterance: Utterance):
        # fast match
        logging.info("Raw Text Sequence: {}".format(utterance.text))
        if utterance.voice in self.fast_match_dict and utterance.text in self.fast_match_dict[utterance.voice]:
            logging.info("Match fast match utterance {}".format(utterance.text))
            syllable_seq, phone_seq = self.fast_match_dict[utterance.voice][utterance.text]
            phone_id_seq, syllable_seq, phone_seq = preprocess_text_to_sequence(self.__hps, phone_seq, syllable_seq)
        else:
            # NOTE(yichao): maybe normalized in ssml, only perform tn in text-mode
            if utterance.normed_text is None:
                utterance.normed_text = self.text_normalizer(utterance.text)

            # NOTE: utterance.normed_text = "摩尔线程" -> "摩#1尔 #2 线#1程" -> "摩尔 #2 线程" -> "摩尔 sp2 线程"
            # -> utterance.normed_text_with_prosody = "摩尔 sp0 线程"
            if self.prosody_predictor is not None:
                utterance.normed_text_with_prosody = self.prosody_predictor.predict(
                        utterance.normed_text,
                        use_sp=True,
                        use_artifact=True)
                if utterance.normed_text_with_prosody != utterance.normed_text:
                    logging.info(f"Prosody predicted for normalized text: {utterance.normed_text}")
                normed_text = utterance.normed_text_with_prosody
            else:
                normed_text = utterance.normed_text

            logging.info("Normalized text sequence: {}".format(normed_text))
            phone_id_seq, syllable_seq, phone_seq = preprocess_text(self.__hps,
                                                                    normed_text,
                                                                    lexicon=self.lexicon,
                                                                    oov_lexicon=self.oov_lexicon)

        utterance.phone_id_seq = phone_id_seq
        utterance.phone_seq = phone_seq
        utterance.syllable_seq = syllable_seq
    
    def synthesize(self, utterances: List[Utterance], speed_rate=1.0, noise_scale=0.667, noise_scale_w=0.8):
        wav_predictions = []
        wav_lengths = []
        for utt in utterances:
            logging.info("For utt_id [{}] with sequence length [{}]".format(utt.utt_id, len(utt.phone_id_seq)))

            phone_id_seq = utt.phone_id_seq
            voice_id = utt.voice
            phone_speed_seq = utt.phone_speed_seq
            if self.__hps.data.add_blank:
                phone_id_seq = commons.intersperse(phone_id_seq, 0)

                # TODO (yi.liu): should we insert 1.0 between phones?
                # phone_speed_seq = commons.intersperse(phone_speed_seq, 1.0) \
                #     if phone_speed_seq is not None else None
                if phone_speed_seq is not None:
                    phone_speed_seq_new  = [1.0]
                    for s in phone_speed_seq:
                        phone_speed_seq_new += [s] * 2
                    phone_speed_seq = phone_speed_seq_new

            phone_id_seq = torch.LongTensor(phone_id_seq)
            x_tst = phone_id_seq.to(self.__device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phone_id_seq.size(0)]).to(self.__device)
            voice_id = torch.LongTensor([voice_id]).to(self.__device)

            phone_speed_seq = torch.FloatTensor(phone_speed_seq).to(self.__device).unsqueeze(0) \
                if phone_speed_seq is not None else None

            with torch.inference_mode(mode=True):
                wav = self.__model.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=voice_id,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=1/speed_rate,
                    length_scale_seq=phone_speed_seq,
                )[0][0,0].data.cpu().float()
                    # noise_scale=.667,
                    # noise_scale_w=0.8,
                wav_predictions.append(wav.expand(1, *wav.shape))
                wav_lengths.append(wav.shape[0])

        wav_predictions = torch.concat(wav_predictions)
        wav_lengths = torch.tensor(wav_lengths)
        wav_predictions = waveform_postprocessing(wav_predictions, wav_lengths)
        for i, utt in enumerate(utterances):
            utt.pred_wav = wav_predictions[i]
            utt.sample_rate = self.__hps.data.sampling_rate

    def http_frontend(self, text, voice, speech_rate=0, pitch_rate=0, energy_rate=0, url="http://172.31.208.11:8082/tts/api/v1/frontend"):
        request_payload = {"text": text, "voice": 'SSB3003',
                           "format": "wav", "sample_rate": 22050, "volume": 50,
                           "speech_rate": speech_rate, "pitch_rate": pitch_rate, "energy_rate": energy_rate}

        resp = requests.post(url, json=request_payload)
        assert resp.status_code == 200, "Error: {}".format(resp.content)
        utterances = []
        content = json.loads(resp.content)
        for i in range(len(content['phone_seqs'])):
            utterance = Utterance("test", text, voice)

            utterance.phone_seq = content['phone_seqs'][i]
            utterance.phone_id_seq, _, _ = preprocess_text_to_sequence(self.__hps, utterance.phone_seq, utterance.phone_seq)

            if 'sos_sils' in content:
                utterance.sos_sil = content['sos_sils'][i]
            if 'eos_sils' in content:
                utterance.eos_sil = content['eos_sils'][i]

            if 'phone_speed_seqs' in content:
                utterance.phone_speed_seq = content['phone_speed_seqs'][i]
            else:
                utterance.phone_speed_seq = [1.0] * len(utterance.phone_seq)

            if 'phone_pitch_seqs' in content:
                utterance.phone_pitch_seq = content['phone_pitch_seqs'][i]
            else:
                utterance.phone_pitch_seq = [1.0] * len(utterance.phone_seq)

            if 'phone_energy_seqs' in content:
                utterance.phone_energy_seq = content['phone_energy_seqs'][i]
            else:
                utterance.phone_energy_seq = [1.0] * len(utterance.phone_seq)

            utterances.append(utterance)

        return utterances


def init_utterances(task_id, gen_req, max_single_utt_length, tts_service, max_sub_utt_length=25):
    utterances = []
    try:
        target_texts = [gen_req.text]
        from_ssml = False
        if gen_req.ssml:
            ssml_texts = parse_ssml(gen_req.ssml, tts_service.text_normalizer)
            logging.info("[{}] parsing SSML text [{}] from [{}]".format(task_id, ssml_texts, gen_req.ssml))
            if ssml_texts:
                target_texts = ssml_texts
                from_ssml = True

        for target_text in target_texts:
            for utt_idx, sub_text in enumerate(preprocess_req_text(target_text, max_length=max_single_utt_length)):
                for sub_utt_idx, sub_sub_text in enumerate(preprocess_req_text(sub_text, max_length=max_sub_utt_length, cut_func=soft_cut_by_comma)):
                    sub_gen_req = copy.deepcopy(gen_req)
                    sub_gen_req.text = sub_sub_text
                    fileid = str(task_id) + "_" + str(utt_idx) + "_" + str(sub_utt_idx)
                    utterance = tts_service.init_utterance(fileid, sub_gen_req.text, sub_gen_req.voice)
                    if from_ssml:
                        # NOTE: ssml parsing and tn is not compatible
                        # todo(yichao): refactoring and fix it
                        utterance.normed_text = utterance.text
                    tts_service.preprocess(utterance)
                    if len(utterance.phone_id_seq) == 0:
                        continue
                    utterances.append(utterance)
    except Exception as e:
        traceback.print_exc()
        logging.error("Error for init utterance for request {}: {}".format(str(gen_req), str(e)))
    return utterances


def unary_synthesize_text(
    tts_service: TTS,
    task_id,
    gen_req: GenerationRequest,
    max_single_utt_length=50,
    use_http_frontend=False,
    service_config=None
):
    start = time.time()
    if use_http_frontend:
        utterances = tts_service.http_frontend(gen_req.text, voice=gen_req.voice)
    else:
        utterances = init_utterances(task_id, gen_req, max_single_utt_length, tts_service, max_sub_utt_length=25)

    for utterance in utterances:
        logging.info("Synthesizing utterance [{}]-[{}] with voice [{}]".format(utterance.utt_id, utterance.text,
                                                                              utterance.voice))
        tts_service.synthesize([utterance], gen_req.speed_rate, gen_req.noise_scale, gen_req.noise_scale_w)

    wav = Utterance.merge_multi_utterances_audio(
        utterances,
        service_config.get("recording_resource_map", None) if service_config is not None else None
    )
    logging.info("synthesize for task_id {} cost {} seconds".format(task_id, time.time() - start))
    return wav


def synthesize_phone_seq(tts_service: TTS, task_id, gen_req: GenerationRequest, max_single_utt_length=50):
    start = time.time()
    speed_rate = gen_req.speed_rate

    utterance = tts_service.init_phone_utterance("debug", gen_req.text, gen_req.voice)
    logging.info("Synthesizing utterance [{}]-[{}] with voice [{}]".format(utterance.utt_id, utterance.text,
                                                                          utterance.voice))
    tts_service.synthesize([utterance], speed_rate)

    wav = Utterance.merge_multi_utterances_audio([utterance])
    logging.info("synthesize for task_id {} cost {} seconds".format(task_id, time.time() - start))
    return wav

