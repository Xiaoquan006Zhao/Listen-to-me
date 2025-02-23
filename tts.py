from kokoro_onnx import Kokoro
import asyncio
import queue
from utils import preprocess_before_generation
import threading
import time
import pygame
import numpy as np
import tempfile
import wave


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()

    def play_np(self, samples, sample_rate):
        assert self.is_audio_playing() is False

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

        self.audioPlayer = AudioPlayer()

        self.stop_event = asyncio.Event()
        self.play_lock = threading.Lock()

    def play_audio(self, samples, sample_rate):
        with self.play_lock:
            if not self.audioPlayer.is_audio_playing():
                self.audioPlayer.play_np(samples, sample_rate)

    async def generate_speech(self, text, voice="am_echo", speed=1.1, lang="en-us"):
        text = preprocess_before_generation(text)
        print("Generating speech")
        print(text)
        stream = self.kokoro.create_stream(text, voice=voice, speed=speed, lang=lang)

        async for samples, sample_rate in stream:
            samples = (samples * 32767).astype("int16")
            print("Playing audio")
            if self.stop_event.is_set():
                break
            self.audio_queue.put((samples, sample_rate))

    async def process_text_queue(self):
        while True:
            print("Processing text queue")
            print("text:", self.text_queue.qsize())
            if not self.text_queue.empty():
                text = self.text_queue.get()
                task = asyncio.create_task(self.generate_speech(text))
                await task
                print("Task completed")
            await asyncio.sleep(2)

    def run_process_text_queue(self):
        asyncio.run(self.process_text_queue())

    async def process_audio_queue(self):
        while True:
            print("Processing audio queue")
            print("audio:", self.audio_queue.qsize())
            if not self.audio_queue.empty() and not self.audioPlayer.is_audio_playing():
                samples, sample_rate = self.audio_queue.get()

                print("GOTcha")
                self.play_audio(samples, sample_rate)
            await asyncio.sleep(2)
            # time.sleep(2)

    def run_process_audio_queue(self):
        asyncio.run(self.process_audio_queue())
        # self.process_audio_queue()

    def add_text_to_queue(self, text):
        self.text_queue.put(text)

    def exit_audio(self):
        with self.play_lock:
            self.audioPlayer.stop_audio()
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def run_process(self):
        text_thread = threading.Thread(target=self.run_process_text_queue)
        audio_thread = threading.Thread(target=self.run_process_audio_queue)
        text_thread.start()
        audio_thread.start()

        return [text_thread, audio_thread]

    def set_stop_event(self):
        self.stop_event.set()


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
    threads = speech_gen.run_process()

    print("Start Loading")
    time.sleep(5)

    speechBuffer = ""
    for c in text:
        speechBuffer += c
        # Buffer until a sentence is complete to generate
        if len(speechBuffer) > 500 and any(speechBuffer.endswith(p) for p in (".", "!", "?", "\n", "。")):
            # threading.Thread(target=self.speechGenerator.start_generate_speech, args=(speechBuffer,)).start()
            speech_gen.add_text_to_queue(speechBuffer)
            speechBuffer = ""

    print("Done")

    for thread in threads:
        thread.join()
