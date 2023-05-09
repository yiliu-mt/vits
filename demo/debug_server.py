import os
import argparse
import uuid
import torch
from flask import Flask
from flask import send_file, request, abort, Response
from scipy.io import wavfile
from tools.logger_util import LOGGER
from service import TTS, GenerationRequest, unary_synthesize_text

app = Flask(__name__, static_url_path='/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dump_wav(wav, sample_rate, basename, path="/tmp"):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, "{}.wav".format(basename))
    print("dump synthesized audio [{}]s to path {}".format(wav.shape[0] / sample_rate, filename))
    wavfile.write(filename, sample_rate, wav)
    return filename


@app.route("/generate", methods=['POST'])
def generate_audio(methods=['POST']):
    if "vits_service" not in app.config:
        abort(Response("the vits service is not initialized...", status=500))
    vits_service = app.config["vits_service"]
    LOGGER.info(request.json)
    request_dict = dict(request.json)

    request_dict.setdefault("format", "wav")
    request_dict.setdefault("noise_scale", 0.667)
    request_dict.setdefault("noise_scale_w", 0.8)

    gen_req = GenerationRequest(
        text=request_dict['text'],
        format=request_dict['format'],
        voice=app.config['speaker_id'],
        noise_scale=request_dict['noise_scale'],
        noise_scale_w=request_dict['noise_scale_w']
    )

    task_id = uuid.uuid4().hex
    audio_data = unary_synthesize_text(
        vits_service,
        task_id,
        gen_req,
        max_single_utt_length=1,
        use_http_frontend=app.config['use_http_frontend'],
    )
    audio_fpath = dump_wav(audio_data, vits_service.sample_rate, task_id)
    LOGGER.info("Return audio fpath [{}]".format(audio_fpath))
    # send back audio
    return send_file(
        audio_fpath,
        mimetype="audio/wav")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_http_frontend", action="store_true", help="If given, use HTTP frontend")
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--lexicon")
    parser.add_argument("--speaker_id", type=int, default=222)
    args = parser.parse_args()

    vits_service = TTS(args.config, args.model, args.lexicon, device)
    app.config["vits_service"] = vits_service
    app.config["use_http_frontend"] = args.use_http_frontend
    app.config["speaker_id"] = args.speaker_id

    LOGGER.info("starting http server...")
    app.run(host='0.0.0.0', debug=False, port=args.port)
