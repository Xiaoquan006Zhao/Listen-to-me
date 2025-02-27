import os
import json
import re
import string
import pyaudio
import numpy as np

CHUNK_DURATION = 0.6  # seconds
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # number of samples per chunk


def record_audio(audio_queue, rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
    """Record audio from the microphone and put chunks into the queue."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk_size)
    print("Recording...")

    try:
        while True:
            audio_data = stream.read(chunk_size)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_queue.put(audio_array)
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def process_hotwords(hotword_file):
    fst_dict = {}
    hotword_msg = ""

    if hotword_file.strip() != "":
        if os.path.exists(hotword_file):
            with open(hotword_file, encoding="utf-8") as f_scp:
                hot_lines = f_scp.readlines()
                for line in hot_lines:
                    words = line.strip().split(" ")
                    if len(words) < 2:
                        print("Please check the format of hotwords")
                        continue
                    try:
                        fst_dict[" ".join(words[:-1])] = int(words[-1])
                    except ValueError:
                        print("Please check the format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = hotword_file

    return hotword_msg


def postprocess_funasr_result(result, remove_punctuation=False):
    text = result[0]["text"]
    text = re.sub(r"<\|.*?\|>", "", text)

    if remove_punctuation:
        punctuation_chars = r"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~。，、；：？！「」『』（）《》【】…—～"
        text = re.sub(f"[{re.escape(punctuation_chars)}]", "", text)

    return text


def remove_markdown(text):
    # Remove headers (e.g., #, ##, ###)
    text = re.sub(r"#{1,6}\s*", "", text)
    # Remove bold and italic markers (e.g., **bold**, *italic*)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    # Remove links (e.g., [text](url))
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    # Remove inline code (e.g., `code`)
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Remove block code (e.g., ```code```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove lists (e.g., -, *, 1.)
    text = re.sub(r"^\s*[\-\*\+]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    # Remove horizontal rules (e.g., ---, ***)
    text = re.sub(r"^\s*[\-\*]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


def preprocess_before_generation(text):
    text = remove_markdown(text)
    text = " ".join(text.split())  # Normalize whitespace
    return text


def emit(socketio, event, data):
    if socketio is not None:
        # print(f"Emitting event: {event} with data: {data}")
        socketio.emit(event, data)


def extract_language_code(s):
    pattern = r"<\|([^|]+)\|>"

    match = re.match(pattern, s)
    if match:
        language = match.group(1)
        return language

    raise RuntimeError("No language code found in the string.")
