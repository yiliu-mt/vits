import argparse
import yaml
import torch
import gradio as gr
import numpy as np
from gradio.inputs import Textbox
from service import TTS, GenerationRequest, unary_synthesize_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradioInfer:
    def __init__(
        self,
        port,
        config,
        model,
        lexicon,
        speaker_id,
        title,
        description,
        article,
        example_inputs
    ):
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs
        self.config = config
        self.model = model
        self.lexicon = lexicon
        self.speaker_id = speaker_id
        self.port = port
        self.vits = TTS(self.config, self.model, self.lexicon, device)
    
    def greet(self, text):
        # Synthesize with long sentence and SSML support
        audio_data = unary_synthesize_text(
            self.vits,
            "test",
            GenerationRequest(text=text, format="wav", voice=self.speaker_id),
            max_single_utt_length=1
        )
        return self.vits.sample_rate, audio_data.astype(np.int16)

    def run(self):
        iface = gr.Interface(fn=self.greet,
                             inputs=[
                                Textbox(lines=5, placeholder=None, default=self.example_inputs[0], label="text")
                             ],
                             outputs="audio",
                             allow_flagging="never",
                             title=self.title,
                             description=self.description,
                             article=self.article,
                             examples=self.example_inputs,
                             examples_per_page=5,
                             enable_queue=True)
        iface.launch(share=False, server_port=self.port, server_name="0.0.0.0")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8911)
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--lexicon")
    parser.add_argument("--speaker_id", type=int, default=222)
    args = parser.parse_args()
    gradio_config = yaml.safe_load(open('gradio/config/gradio_settings.yaml'))
    g = GradioInfer(
        port=args.port,
        config=args.config,
        model=args.model,
        lexicon=args.lexicon,
        speaker_id=args.speaker_id,
        **gradio_config)
    g.run()
