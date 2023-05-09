import base64
import sys
import tempfile

import soundfile

sys.path.append("./")

import json
import traceback
from collections import OrderedDict

from flask import Flask, send_from_directory
from flask import send_file, request, jsonify, abort, Response
from flask_cors import CORS
import uuid
import csv

import jieba
from flask import render_template
from websocket_server import WebsocketServer
import threading

from demo.servlet import unary_synthesize_text, read_speaker_to_model, \
    streaming_synthesize_text, generate_audio_meta, generate_audio_chunk, generate_audio_end_message
from serving.tts_service import GenerationRequest
from tools.logger_util import LOGGER
from utils.tools import dump_wav
from serving.grpc_servlet import TTSServlet

app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

@app.route('/tts/api/v1/<path:path>')
def send_static_file(path):
    return send_from_directory('./', path)

@app.route("/g2p", methods=['POST'])
def g2p():
    if "tts_service" not in app.config:
        abort(Response("the tts service is not initialized...", status=500))
    tts_service = app.config["tts_service"]
    request_dict = dict(request.json)
    max_single_utt_length = int(request_dict.pop("max_single_utt_length", 1))
    max_sub_utt_length = int(request_dict.pop("max_sub_utt_length", 25))
    if "version" not in request_dict:
        request_dict["version"] = DEFAULT_VERSION
    if 'speaker' in request_dict:
        request_dict['voice'] = request_dict['speaker']
    if 'format' not in request_dict:
        request_dict['format'] = "wav"
    gen_req = GenerationRequest(**request_dict)
    speaker_table_fpath = "./demo/speaker_table_{}.txt".format(gen_req.version)
    speaker2model = read_speaker_to_model(speaker_table_fpath)
    assert gen_req.voice in speaker2model, "voice {} is not in current version".format(gen_req.voice)
    tts = tts_service.voice_name2model[gen_req.voice]
    task_id = uuid.uuid4().hex
    utterances = tts.prepare_utterances(task_id, gen_req.text, gen_req.ssml, gen_req.voice, max_single_utt_length, max_sub_utt_length=max_sub_utt_length)
    phone_seq = []
    syllable_seq = []
    token_seq = []
    text = []
    for utt in utterances:
        phone_seq.append(utt.phone_seq)
        syllable_seq.append(utt.syllable_seq)
        token_seq.append(list(filter(lambda x: len(x.strip()) > 0, list(jieba.cut(utt.normed_text if utt.normed_text else utt.text)))))
        text.append(utt.text)
    return jsonify({"phone_seq": phone_seq, "syllable_seq": syllable_seq, "token_seq": token_seq, "text": text})


@app.route("/tts/api/v1/generate", methods=['POST'])
def generate_audio(methods=['POST']):
    if "tts_service" not in app.config:
        abort(Response("the tts service is not initialized...", status=500))
    tts_service = app.config["tts_service"]
    # get request info
    LOGGER.info(request.json)
    request_dict = dict(request.json)
    if 'speaker' in request_dict:
        request_dict['voice'] = request_dict['speaker']
    if 'format' not in request_dict:
        request_dict['format'] = "wav"
    gen_req = GenerationRequest(**request_dict)
    if gen_req.voice not in tts_service.voice_name2model:
        abort(Response("voice {} is not in current version".format(gen_req.voice), status=400))
    task_id = uuid.uuid4().hex
    tts = tts_service.voice_name2model[gen_req.voice]
    audio_data = unary_synthesize_text(task_id, tts, gen_req, max_single_utt_length=1)
    if gen_req.format == "pcm":
        audio_data_bytes = audio_data.tobytes()
        return json.dumps({
            "audio": base64.b64encode(audio_data_bytes).decode()
        })
    else:
        audio_fpath = dump_wav(audio_data, tts.sample_rate, task_id)
        LOGGER.info("Return audio fpath [{}]".format(audio_fpath))
        # send back audio
        return send_file(
            audio_fpath,
            mimetype="audio/wav")


def receive_synthesize_request(client, server, message):
    try:
        tts_service = app.config["tts_service"]
        req_msg = json.loads(message)
        LOGGER.info("receive websocket request message {}".format(req_msg))
        task_id = "tts_websocket_" + req_msg["header"]["task_id"]
        req_payload = req_msg["payload"]
        text = req_payload['text']
        voice = req_payload['voice']
        volume = req_payload['volume']
        speech_rate = req_payload['speech_rate']
        pitch_rate = req_payload['pitch_rate']
        energy_rate = req_payload['energy_rate']
        enable_subtitle = req_payload['enable_subtitle']
        enable_interval = req_payload['enable_interval']
        format = req_payload.get('format', 'pcm')
        version = req_payload.get("version", DEFAULT_VERSION)
        ssml = req_payload.get("ssml", "")
        gen_req = GenerationRequest(text=text, ssml=ssml, voice=voice, version=version,
                                    format=format, pitch_rate=pitch_rate, energy_rate=energy_rate,
                                    speech_rate=speech_rate, volume=volume)


        def transmit_audio_meta_callback(total_chunk, duration, intervals):
            LOGGER.info("The duration of audio in task {}: {}".format(task_id, duration))
            audio_meta = generate_audio_meta(task_id, total_chunk, duration, intervals, gen_req.text)
            server.send_message(client, json.dumps(audio_meta))
            LOGGER.info("Send back audio meta for task_id {}".format(task_id))

        def transmit_audio_callback(chunk_idx, audio_data):
            if format == "pcm":
                audio_data_bytes = audio_data.tobytes()
            elif format == "wav":
                LOGGER.info("Convert pcm to wav...")
                wav_fpath = tempfile.NamedTemporaryFile().name + ".wav"
                soundfile.write(wav_fpath, audio_data, 22050, 'PCM_16')
                audio_data_bytes = open(wav_fpath, "rb").read()
            else:
                raise NotImplementedError
            audio_chunk = generate_audio_chunk(task_id, chunk_idx, audio_data_bytes)
            server.send_message(client, json.dumps(audio_chunk))
            LOGGER.info("Send back audio chunk {} with {} samples".format(chunk_idx, len(audio_data)))

        def transmit_audio_end_callback():
            server.send_message(client, json.dumps(generate_audio_end_message(task_id)))
            LOGGER.info("Send back audio end message for task_id {}".format(task_id))

        assert gen_req.voice in tts_service.voice_name2model, "voice [{}] is not supported in version {}".format(
            gen_req.voice, DEFAULT_VERSION)
        tts = tts_service.voice_name2model[gen_req.voice]
        streaming_synthesize_text(task_id,  tts, gen_req, max_single_utt_length=1,
                                  transmit_audio_meta_callback=transmit_audio_meta_callback,
                                  transmit_audio_callback=transmit_audio_callback,
                                  transmit_audio_end_callback=transmit_audio_end_callback,
                                  auto_chunking_duration=0.2)

    except Exception as e:
        traceback.print_exc()
        server.send_message(client, "error {}".format(e))
    server.disconnect_clients_gracefully()


def read_speaker_table(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        rows = list(reader)

    speaker_table = OrderedDict()
    for ds_source, speaker_id, gender, nickname, email, model_version in rows:
        speaker_table[speaker_id] = [nickname, gender, ds_source]
    return speaker_table


def read_default_speaker_name(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        rows = list(reader)
    return rows[0][1]


@app.route("/tts/api/v1/voices", methods=["GET"])
def get_voices():
    voice_list = json.load(open("demo/voices.json"))
    return json.dumps(voice_list)


@app.route("/v2", methods=["GET"])
def main_v2_handler():
    """Landing select speaker portal page."""
    mode_version = request.args.get("version", DEFAULT_VERSION)
    if mode_version != DEFAULT_VERSION:
        return "Version {} is not available!!!".format(mode_version)
    spk_table_txt_fpath = "./demo/speaker_table_{}.txt".format(mode_version)
    speaker_table = read_speaker_table(spk_table_txt_fpath)
    default_speaker = read_default_speaker_name(spk_table_txt_fpath)
    return render_template(
        'portal_select.html',
        title="Simple TTS Demo",
        description="",
        version=mode_version,
        speaker_table=speaker_table,
        default_speaker_id=default_speaker,
    )


if __name__ == '__main__':
    _, default_version, target_voices = sys.argv
    DEFAULT_VERSION = default_version
    assert target_voices, "target_voices is empty"
    target_voices = target_voices.split(",")

    _tts_service = TTSServlet.load_by_version(DEFAULT_VERSION, target_voices=target_voices)
    app.config["tts_service"] = _tts_service

    ws_server = WebsocketServer(host='0.0.0.0', port=8089)
    ws_server.set_fn_message_received(receive_synthesize_request)

    LOGGER.info("starting http server...")
    threading.Thread(target=app.run, kwargs={"host": '0.0.0.0', "debug": False, "port": 8081}).start()
    # app.run(host='0.0.0.0', debug=False, port=8081)
    LOGGER.info("starting ws server...")
    ws_server.run_forever()
