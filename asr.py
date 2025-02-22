import pyaudio
import numpy as np
import threading
import queue
from funasr import AutoModel
from enum import Enum, auto
import os

audio_queue = queue.Queue()  # Raw audio chunks from microphone


class State(Enum):
    ONLINE = auto()
    OFFLINE = auto()
    IDLE = auto()


class SpeechRecognizer:
    def __init__(
        self,
        RATE=16000,
        CHUNK=9600,
        CHUNK_SIZE=[0, 10, 4],
        ENCODER_CHUNK_LOOK_BACK=2,
        DECODER_CHUNK_LOOK_BACK=0,
        DISABLE=True,
    ):
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

        self.RATE = RATE
        self.CHUNK = CHUNK
        self.CHUNK_SIZE = CHUNK_SIZE
        self.ENCODER_CHUNK_LOOK_BACK = ENCODER_CHUNK_LOOK_BACK
        self.DECODER_CHUNK_LOOK_BACK = DECODER_CHUNK_LOOK_BACK

        self.cache = {}
        self.accumulated_speech = []

        self.state = State.IDLE

        self.is_ending_counter = 0
        self.is_ending_counter_threshold = 2

        self.is_idle_counter = 0
        self.is_idle_counter_threshold = 8

        self.accumulated_speech_threshold = 50
        self.text_2pass_offline = ""
        self.text_2pass_online = ""

    def process_audio_chunk(self, audio_data):
        """Process a single audio chunk using VAD and ASR models."""
        if audio_data is None:
            return

        vad_res = self.vad_model.generate(input=audio_data)

        # Online
        if len(vad_res[0]["value"]) > 0:
            self.state = State.ONLINE

            online_res = self.online_model.generate(
                input=audio_data,
                cache=self.cache,
                is_final=False,
                chunk_size=self.CHUNK_SIZE,
                encoder_chunk_look_back=self.ENCODER_CHUNK_LOOK_BACK,
                decoder_chunk_look_back=self.DECODER_CHUNK_LOOK_BACK,
            )
            self.text_2pass_online += "{}".format(online_res[0]["text"])
            self.accumulated_speech.append(audio_data)
            self.reset_flags()

        # Offline
        if (self.state == State.ONLINE and len(vad_res[0]["value"]) == 0) or len(
            self.accumulated_speech
        ) > self.accumulated_speech_threshold:
            self.is_ending_counter += 1
            if (
                self.is_ending_counter >= self.is_ending_counter_threshold
                or len(self.accumulated_speech) > self.accumulated_speech_threshold
            ):
                self.state = State.OFFLINE
                self.process_accumulated_speech()
                self.reset_flags()

        # Idle
        if not self.state == State.IDLE and len(vad_res[0]["value"]) == 0:
            self.is_idle_counter += 1
            if self.is_idle_counter >= self.is_idle_counter_threshold:
                self.state = State.IDLE
                self.reset_flags()

    def process_accumulated_speech(self):
        """Process accumulated speech using the offline model."""
        offline_res = self.offline_model.generate(
            input=np.concatenate(self.accumulated_speech), batch_size_s=300, batch_size_threshold_s=60
        )
        self.text_2pass_online = ""
        self.text_2pass_offline += "{}".format(offline_res[0]["text"])

    def reset_flags(self):
        if self.state == State.IDLE:
            self.is_idle_counter = 0
        elif self.state == State.OFFLINE:
            self.accumulated_speech = []
            self.state = State.OFFLINE
            self.is_ending_counter = 0
            self.is_idle_counter = 0
            self.cache = {}
            self.update_display()
        elif self.state == State.ONLINE:
            self.is_ending_counter = 0
            self.is_idle_counter = 0
            self.update_display()

    def get_transcription(self):
        return self.text_2pass_offline + self.text_2pass_online

    # For downstream processing to indicate that task has been completed
    def reset_external_transcription(self):
        self.text_2pass_offline = ""
        self.text_2pass_online = ""

    def update_display(self):
        # os.system("clear")
        print("Text print:", self.get_transcription())


def record_audio():
    """Record audio from the microphone and put chunks into the queue."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=9600)
    print("Recording...")

    try:
        while True:
            audio_data = stream.read(9600)
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
