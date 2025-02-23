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


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()

    def play_np(self, samples, sample_rate):
        if self.is_audio_playing():
            raise RuntimeError("Audio is already playing")

        if samples.ndim == 1:
            samples = np.column_stack((samples, samples))

        temp_file = self._save_to_wav(samples, sample_rate)
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

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
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.audio_player = AudioPlayer()
        self.stop_event = asyncio.Event()
        self.is_kokoro_running = False

    def play_audio(self, samples, sample_rate):
        self.audio_player.play_np(samples, sample_rate)

    async def generate_speech(self, text, voice="am_echo", speed=1.1, lang="en-us"):
        text = preprocess_before_generation(text)
        stream = self.kokoro.create_stream(text, voice=voice, speed=speed, lang=lang)

        async for samples, sample_rate in stream:
            samples = (samples * 32767).astype("int16")
            if self.stop_event.is_set():
                break
            self.audio_queue.put((samples, sample_rate))

    async def process_text_queue(self):
        while not self.stop_event.is_set():
            if not self.text_queue.empty():
                text = self.text_queue.get()
                self.is_kokoro_running = True
                await self.generate_speech(text)
                self.is_kokoro_running = False
            await asyncio.sleep(0.1)

    async def process_audio_queue(self):
        while not self.stop_event.is_set():
            if not self.audio_queue.empty() and not self.audio_player.is_audio_playing():
                samples, sample_rate = self.audio_queue.get()
                self.play_audio(samples, sample_rate)
            await asyncio.sleep(0.1)

    def add_text_to_queue(self, text):
        self.text_queue.put(text)

    def run(self):
        async def main():
            text_task = asyncio.create_task(self.process_text_queue())
            audio_task = asyncio.create_task(self.process_audio_queue())
            await asyncio.gather(text_task, audio_task)

        asyncio.run(main())

    def interrupt(self):
        self.clear_queues()
        self.audio_player.stop_audio()
        self.stop_event.set()

    def clear_queues(self):
        while not self.text_queue.empty():
            self.text_queue.get()
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def restart(self):
        self.clear_queues()
        self.stop_event.clear()
        return threading.Thread(target=self.run)


# Example usage
if __name__ == "__main__":
    text = """
    Yes, there are many reasons why people might feel sad at times. Sadness is a natural human emotion, and like all emotions, it serves an important purpose. It can signal that something in our lives isn't as we wish it to be, prompting us to take action or seek comfort. Here are several factors that can lead to feelings of sadness:

    1. **Major Life Events**: Significant life changes such as the death of a loved one, the end of a relationship, job loss, or moving to a new place can all trigger sadness.

    2. **Personal Challenges**: Dealing with personal struggles like health issues, financial difficulties, or unmet personal goals can cause feelings of sadness.

    3. **Mental Health Disorders**: Conditions such as depression, anxiety, bipolar disorder, and post-traumatic stress disorder (PTSD) can include episodes of sadness as a primary symptom.

    4. **Chemical Imbalances**: The body's complex system of neurotransmitters, including serotonin and dopamine, can affect mood and lead to feelings of sadness if the balance is off.
    """

    speech_gen = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
    speech_thread = threading.Thread(target=speech_gen.run)
    speech_thread.start()

    speech_buffer = ""
    for c in text:
        speech_buffer += c
        if len(speech_buffer) > 50 and any(speech_buffer.endswith(p) for p in (".", "!", "?", "\n", "。")):
            speech_gen.add_text_to_queue(speech_buffer)
            speech_buffer = ""

    speech_thread.join()
