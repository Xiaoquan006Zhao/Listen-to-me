import os
import json


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


def record_audio(audio_queue, RATE=16000, CHUNK=9600):
    import pyaudio
    import numpy as np

    """Record audio from the microphone and put chunks into the queue."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording...")

    try:
        while True:
            audio_data = stream.read(CHUNK)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            audio_queue.put(audio_array)

    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        stream.stop_stream()
        stream.close()
