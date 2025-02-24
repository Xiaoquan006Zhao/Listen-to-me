import asyncio
import queue
import threading
import time
import pygame
import numpy as np
import tempfile
import wave
from kokoro_onnx import Kokoro
from utils import preprocess_before_generation
import os
from enum import Enum, auto


class SpeechGeneratorTask(Enum):
    TEXT = auto()
    AUDIO = auto()


class AudioPlayer:
    def __init__(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    async def play_np(self, samples, sample_rate):
        if self.is_audio_playing():
            raise RuntimeError("Audio is already playing")

        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a NumPy array")

        if samples.ndim == 1:
            samples = np.column_stack((samples, samples))  # Convert mono to stereo

        temp_file = self._save_to_wav(samples, sample_rate)
        try:
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            duration = pygame.mixer.Sound(temp_file).get_length()
            await asyncio.sleep(duration)
        finally:
            os.remove(temp_file)

    def stop_audio(self):
        if self.is_audio_playing():
            pygame.mixer.music.stop()

    def is_audio_playing(self):
        return pygame.mixer.music.get_busy()

    def _save_to_wav(self, samples, sample_rate):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        with wave.open(temp_filename, "wb") as wf:
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 16-bit samples
            wf.setframerate(sample_rate)
            wf.writeframes(samples.astype(np.int16).tobytes())
        return temp_filename


class SpeechGenerator:
    def __init__(self, model_path, voices_path):
        self.kokoro = Kokoro(model_path, voices_path)
        self.audio_player = AudioPlayer()
        self.loop = None

        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()

        self.stop_event = asyncio.Event()
        self.task_completed_event = threading.Event()

        self.text_task = None
        self.audio_task = None
        self.tasks = []

    async def play_audio(self, samples, sample_rate):
        await self.audio_player.play_np(samples, sample_rate)

    async def generate_speech(self, text, voice="am_echo", speed=1.2, lang="en-us"):
        text = preprocess_before_generation(text)
        stream = self.kokoro.create_stream(text, voice=voice, speed=speed, lang=lang)

        async for samples, sample_rate in stream:
            samples = (samples * 32767).astype("int16")
            if self.stop_event.is_set():
                break
            self.audio_queue.put((samples, sample_rate))
            self.create_task("audio_task", self.process_audio_queue)

    async def process_text_queue(self):
        while not self.text_queue.empty() and not self.stop_event.is_set():
            text = self.text_queue.get()
            await self.generate_speech(text)

    async def process_audio_queue(self):
        while not self.audio_queue.empty() and not self.stop_event.is_set():
            if not self.audio_player.is_audio_playing():
                samples, sample_rate = self.audio_queue.get()
                await self.play_audio(samples, sample_rate)

    def add_text_to_queue(self, text):
        self.text_queue.put(text)
        self.create_task("text_task", self.process_text_queue)

    def create_task(self, task_attr, process_func):
        if self.loop is not None and (getattr(self, task_attr) is None or getattr(self, task_attr).done()):
            concurrent_future = asyncio.run_coroutine_threadsafe(process_func(), self.loop)
            task = asyncio.wrap_future(concurrent_future, loop=self.loop)
            setattr(self, task_attr, task)
            self.tasks.append(task)

    def run(self):
        async def main():
            tasks_started = False

            while not self.stop_event.is_set() and not (len(self.tasks) == 0 and tasks_started):
                if len(self.tasks) > 0:
                    tasks_started = True
                    done, _ = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        self.tasks.remove(task)

            self.interrupt()

        if not self.loop:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        asyncio.run_coroutine_threadsafe(main(), self.loop)

    def interrupt(self):
        self.clear_queues()
        self.stop_event.set()
        self.audio_player.stop_audio()
        self.task_completed_event.set()

    def clear_queues(self):
        while not self.text_queue.empty():
            self.text_queue.get()
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def restart(self):
        self.clear_queues()
        self.stop_event.clear()
        self.task_completed_event.clear()
        self.text_task = None
        self.audio_task = None
        self.tasks = []
        event_loop_thread = threading.Thread(target=self.run)
        return event_loop_thread


# Example usage
if __name__ == "__main__":
    text = """
    It was a bright cold day in April, and the clocks were striking thirteen.
Winston Smith, his chin nuzzled into his breast in an effort to escape the
vile wind, slipped quickly through the glass doors of Victory Mansions,
though not quickly enough to prevent a swirl of gritty dust from entering
along with him.
    """

    speech_gen = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
    speech_thread = speech_gen.restart()
    speech_thread.start()

    speech_buffer = ""
    for c in text:
        speech_buffer += c
        if len(speech_buffer) > 5 and any(speech_buffer.endswith(p) for p in (".", "!", "?", "\n", "。")):
            speech_gen.add_text_to_queue(speech_buffer)
            speech_buffer = ""

    speech_thread.join()
