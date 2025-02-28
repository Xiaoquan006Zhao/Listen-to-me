from text_to_speech.ttsInterface import ITTSModel
from kokoro_onnx import Kokoro


class kokoroModel(ITTSModel):
    def __init__(self, model_path, voices_path):
        self.model = Kokoro(model_path, voices_path)
        self.voice = "am_echo"
        self.speed = 1.3
        self.language = "en-us"

    def synthesize(self, text):
        # Stream audio creation asynchronously in the background, yielding chunks as they are processed.
        return self.model.create_stream(text, voice=self.voice, speed=self.speed, lang=self.language)
