import pyaudio
import numpy as np
import threading
import queue
from funasr import AutoModel
import os

# Constants
RATE = 16000  # Sampling rate (16kHz)
CHUNK = 9600  # Number of frames per buffer
CHUNK_SIZE = [0, 10, 4]  # Streaming model chunk size
ENCODER_CHUNK_LOOK_BACK = 2  # Lookback for encoder self-attention
DECODER_CHUNK_LOOK_BACK = 0  # Lookback for decoder cross-attention
DISABLE = True  # Disable logs and progress bars

# Queues
audio_queue = queue.Queue()  # Raw audio chunks from microphone
offline_audio_queue = queue.Queue()  # Accumulated results for offline model


class SpeechRecognizer:
    def __init__(self):
        """Initialize ASR models and variables."""
        self.vad_model = AutoModel(model="fsmn-vad", disable_log=DISABLE, disable_pbar=DISABLE)
        self.online_model = AutoModel(model="paraformer-zh-streaming", disable_log=DISABLE, disable_pbar=DISABLE)
        self.offline_model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="ct-punc",
            disable_log=DISABLE,
            disable_pbar=DISABLE,
        )
        self.cache = {}
        self.accumulated_speech = []
        self.is_speech_ongoing = False
        self.is_ending_counter = 0
        self.accumulated_speech_threshold = 50
        self.is_ending_counter_threshold = 2
        self.text_print_2pass_offline = ""
        self.text_print_2pass_online = ""

    def process_audio_chunk(self, audio_data):
        """Process a single audio chunk using VAD and ASR models."""
        if audio_data is None:
            return

        vad_res = self.vad_model.generate(input=audio_data)
        if len(vad_res[0]["value"]) > 0:
            self.is_speech_ongoing = True
            online_res = self.online_model.generate(
                input=audio_data,
                cache=self.cache,
                is_final=False,
                chunk_size=CHUNK_SIZE,
                encoder_chunk_look_back=ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=DECODER_CHUNK_LOOK_BACK,
            )
            self.text_print_2pass_online += "{}".format(online_res[0]["text"])
            self.accumulated_speech.append(audio_data)
            self.is_ending_counter = 0

            self.update_display()

        if (self.is_speech_ongoing and len(vad_res[0]["value"]) == 0) or len(
            self.accumulated_speech
        ) > self.accumulated_speech_threshold:
            self.is_ending_counter += 1
            if (
                self.is_ending_counter >= self.is_ending_counter_threshold
                or len(self.accumulated_speech) > self.accumulated_speech_threshold
            ):
                self.process_accumulated_speech()
                self.reset_flags()

                self.update_display()

    def process_accumulated_speech(self):
        """Process accumulated speech using the offline model."""
        offline_res = self.offline_model.generate(
            input=np.concatenate(self.accumulated_speech), batch_size_s=300, batch_size_threshold_s=60
        )
        self.text_print_2pass_online = ""
        self.text_print_2pass_offline += "{}".format(offline_res[0]["text"])

    def reset_flags(self):
        """Reset flags and accumulated speech."""
        self.accumulated_speech = []
        self.is_speech_ongoing = False
        self.is_ending_counter = 0
        self.cache = {}

    def get_transcription(self):
        """Return the combined transcription."""
        return self.text_print_2pass_offline + self.text_print_2pass_online

    def update_display(self):
        """Update the display with the current transcription."""
        os.system("clear")
        print("Text print:", self.get_transcription())


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
    """Process audio chunks from the queue."""
    recognizer = SpeechRecognizer()

    while True:
        audio_data = audio_queue.get()
        if audio_data is None:
            break

        recognizer.process_audio_chunk(audio_data)
        recognizer.update_display()


def main():
    """Main function to start recording and processing threads."""
    record_thread = threading.Thread(target=record_audio)
    record_thread.start()

    try:
        process_audio()
    except KeyboardInterrupt:
        print("Processing stopped.")
    finally:
        audio_queue.put(None)
        record_thread.join()


if __name__ == "__main__":
    main()
