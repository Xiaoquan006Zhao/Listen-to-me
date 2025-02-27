import pyaudio
import numpy as np
import threading
import queue
from funasr import AutoModel
from enum import Enum, auto
import os
from modelscope.pipelines import pipeline
from utils import postprocess_funasr_result, emit, extract_language_code

audio_queue = queue.Queue()  # Raw audio chunks from microphone


class SpeechRecognizerState(Enum):
    ONLINE = auto()
    OFFLINE = auto()
    IDLE = auto()


class SpeechRecognizer:
    def __init__(
        self,
        socketio=None,
        RATE=16000,
        CHUNK=9600,
        DISABLE=True,
        verify_speaker_threshold=0.35,
        accumulated_speech_threshold=50,
        is_ending_counter_threshold=2,
        is_idle_counter_threshold=8,
    ):
        self.vad_model = AutoModel(model="fsmn-vad", disable_log=DISABLE, disable_pbar=DISABLE, disable_update=DISABLE)

        self.online_model = self.offline_model = AutoModel(
            model="FunAudioLLM/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            hub="hf",
            disable_log=DISABLE,
            disable_pbar=DISABLE,
            disable_update=DISABLE,
        )

        self.RATE = RATE
        self.CHUNK = CHUNK

        self.online_cache = {}
        self.accumulated_speech = []

        self.listening_to_user_event = threading.Event()

        self.state = SpeechRecognizerState.IDLE

        self.is_ending_counter = 0
        self.is_ending_counter_threshold = is_ending_counter_threshold

        self.is_idle_counter = 0
        self.is_idle_counter_threshold = is_idle_counter_threshold

        self.accumulated_speech_threshold = accumulated_speech_threshold
        self.text_2pass_offline = ""
        self.text_2pass_online = ""

        self.sv_pipeline = pipeline(task="speaker-verification", model="iic/speech_campplus_sv_zh-cn_16k-common")
        self.initial_speaker = None
        self.speaker_verified = False
        self.verify_speaker_threshold = verify_speaker_threshold

        self.socketio = socketio
        self.current_language = None

    def process_audio_chunk(self, audio_data):
        """Process a single audio chunk using VAD and ASR models."""
        if audio_data is None:
            return

        emit(self.socketio, "user_idle_counter", {"counter": self.is_idle_counter_threshold - self.is_idle_counter})

        vad_res = self.vad_model.generate(input=audio_data)

        self.speaker_verified = self.verify_speaker(audio_data)

        user_speaking = len(vad_res[0]["value"]) > 0 and self.speaker_verified

        # Online
        if user_speaking:
            online_res = self.online_model.generate(
                input=audio_data,
                cache=self.online_cache,
                language=self.current_language if self.current_language is not None else "auto",
                use_itn=True,
                batch_size_s=60,
            )

            online_transcription = postprocess_funasr_result(online_res, remove_punctuation=True)
            emit(self.socketio, "online_transcription", {"message": online_transcription})

            self.text_2pass_online += online_transcription
            self.accumulated_speech.append(audio_data)
            self.set_state(SpeechRecognizerState.ONLINE)

        # Offline
        if (self.state == SpeechRecognizerState.ONLINE and not user_speaking) or len(
            self.accumulated_speech
        ) > self.accumulated_speech_threshold:
            self.is_ending_counter += 1
            if (
                self.is_ending_counter >= self.is_ending_counter_threshold
                or len(self.accumulated_speech) > self.accumulated_speech_threshold
            ):
                self.process_accumulated_speech()
                self.set_state(SpeechRecognizerState.OFFLINE)

        # Idle
        if not self.state == SpeechRecognizerState.IDLE and not user_speaking:
            self.is_idle_counter += 1
            if self.is_idle_counter >= self.is_idle_counter_threshold:
                self.set_state(SpeechRecognizerState.IDLE)

    def process_accumulated_speech(self):
        """Process accumulated speech using the offline model."""
        audio_data = np.concatenate(self.accumulated_speech)
        self.speaker_verified = False
        if self.initial_speaker is None:
            self.initial_speaker = audio_data

        offline_res = self.offline_model.generate(
            input=audio_data,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )

        self.current_language = extract_language_code(offline_res[0]["text"])
        offline_transcription = postprocess_funasr_result(offline_res)

        self.text_2pass_online = ""
        self.text_2pass_offline += offline_transcription
        emit(self.socketio, "offline_transcription", {"message": offline_transcription})

    def reset_flags(self):
        if self.state == SpeechRecognizerState.IDLE:
            self.is_idle_counter = 0
        elif self.state == SpeechRecognizerState.OFFLINE:
            self.accumulated_speech = []
            self.is_ending_counter = 0
            self.is_idle_counter = 0
            self.online_cache = {}
            self.update_display()
        elif self.state == SpeechRecognizerState.ONLINE:
            self.is_ending_counter = 0
            self.is_idle_counter = 0
            self.update_display()

    def verify_speaker(self, audio_data):
        if self.initial_speaker is None:
            return True
        else:
            sv_res = self.sv_pipeline([audio_data, self.initial_speaker], thr=self.verify_speaker_threshold)
            print("Speaker verification result:", sv_res)
            if sv_res["text"] == "yes":
                return True

        return False

    def get_transcription(self):
        return self.text_2pass_offline + self.text_2pass_online

    # For downstream processing to indicate that task has been completed
    def reset_external_transcription(self):
        self.text_2pass_offline = ""
        self.text_2pass_online = ""
        emit(self.socketio, "reset_transcription", {"reset": True})

    def update_display(self):
        print("Text print:", self.get_transcription())

    def set_state(self, state):
        self.state = state
        self.reset_flags()
        if self.state != SpeechRecognizerState.IDLE:
            self.listening_to_user_event.set()
            emit(self.socketio, "listening_to_user", {"listening": True})
        else:
            self.listening_to_user_event.clear()
