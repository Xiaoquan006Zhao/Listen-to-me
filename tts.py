import asyncio
import threading
import base64
from kokoro_onnx import Kokoro
from utils import preprocess_before_generation, emit
import queue


class SpeechGenerator:

    def __init__(
        self,
        model_path,
        voices_path,
        interrupt_event,
        socketio=None,
        buffer_threshold=50,
    ):
        self.kokoro = Kokoro(model_path, voices_path)
        self.socketio = socketio
        self.buffer_threshold = buffer_threshold

        self.text_buffer = ""
        self.text_queue = asyncio.Queue()
        self.interrupt_event = interrupt_event

        self.loop = None

        def run_loop():
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()

        threading.Thread(target=run_loop).start()

    async def generate_speech(self, text, voice="am_echo", speed=1.3, lang="en-us"):
        text = preprocess_before_generation(text)
        stream = self.kokoro.create_stream(text, voice=voice, speed=speed, lang=lang)
        async for samples, sample_rate in stream:
            if self.interrupt_event.is_set():
                break
            # Convert samples to int16 PCM format.
            samples = (samples * 32767).astype("int16")
            audio_base64 = base64.b64encode(samples.tobytes()).decode("utf-8")
            emit(self.socketio, "audio_stream", {"samplerate": sample_rate, "samples": audio_base64, "stopped": False})

    async def process_queue(self):
        while not self.interrupt_event.is_set():
            text = await self.text_queue.get()
            await self.generate_speech(text)
            self.text_queue.task_done()

        self.stop()

    def add_text(self, text, buffered=True):
        self.text_buffer += text
        if self.text_buffer and (
            not buffered or (len(self.text_buffer) >= self.buffer_threshold and self.text_buffer[-1] in " ,:;.!?\n")
        ):
            asyncio.run_coroutine_threadsafe(self.text_queue.put(self.text_buffer), self.loop)
            self.text_buffer = ""

    def stop(self):
        while not self.text_queue.empty():
            self.text_queue.get()
        self.text_buffer = ""

    def start(self):
        asyncio.run_coroutine_threadsafe(self.process_queue(), self.loop)
