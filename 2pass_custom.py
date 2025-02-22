import pyaudio
import numpy as np
import threading
import queue
from funasr import AutoModel
import os

RATE = 16000  # Sampling rate (16kHz)
CHUNK = 9600  # Number of frames per buffer
chunk_size = [0, 10, 4]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 2  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 0  # number of encoder chunks to lookback for decoder cross-attention

disable_pbar = disable_log = DISABLE = True
vad_model = AutoModel(model="fsmn-vad", disable_log=DISABLE, disable_pbar=DISABLE)

online_model = AutoModel(model="paraformer-zh-streaming", disable_log=DISABLE, disable_pbar=DISABLE)

offline_model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 60000},
    punc_model="ct-punc",
    disable_log=DISABLE,
    disable_pbar=DISABLE,
)

# Queues to hold audio chunks and results
audio_queue = queue.Queue()  # Raw audio chunks from microphone
offline_audio_queue = queue.Queue()  # Accumulated results for offline model


def record_audio():
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


def process_audio():
    is_speech_ongoing = False
    accumulated_speech = []
    accumulated_speech_threshold = 50
    is_ending_counter = 0
    is_ending_counter_threshold = 3
    cache = {}

    text_print_2pass_offline = ""
    text_print_2pass_online = ""
    text_print = ""

    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            continue

        speech_chunk = audio_data

        vad_res = vad_model.generate(input=speech_chunk)
        # print("VAD:", vad_res)

        if len(vad_res[0]["value"]) > 0:
            is_speech_ongoing = True
            online_res = online_model.generate(
                input=speech_chunk,
                cache=cache,
                is_final=False,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )
            # print("Online Model:", online_res)

            text_print_2pass_online += "{}".format(online_res[0]["text"])
            text_print = text_print_2pass_offline + text_print_2pass_online

            accumulated_speech.append(speech_chunk)
            is_ending_counter = 0

        if (is_speech_ongoing and len(vad_res[0]["value"]) == 0) or len(
            accumulated_speech
        ) > accumulated_speech_threshold:
            is_ending_counter += 1
            if (
                is_ending_counter >= is_ending_counter_threshold
                or len(accumulated_speech) > accumulated_speech_threshold
            ):
                offline_res = offline_model.generate(
                    input=np.concatenate(accumulated_speech), batch_size_s=300, batch_size_threshold_s=60
                )
                # print("Offline Model:", offline_res)
                text_print_2pass_online = ""
                text_print_2pass_offline += "{}".format(offline_res[0]["text"])
                text_print = text_print_2pass_offline

                # Reset flags
                accumulated_speech = []
                is_speech_ongoing = False
                is_ending_counter = 0
                cache = {}

        os.system("clear")
        print("Text print:", text_print)


record_thread = threading.Thread(target=record_audio)
record_thread.start()

try:
    process_audio()
except KeyboardInterrupt:
    print("Processing stopped.")
finally:
    audio_queue.put(None)
    record_thread.join()
    # audio.terminate()
