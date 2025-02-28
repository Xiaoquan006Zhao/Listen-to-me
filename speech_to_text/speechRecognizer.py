import numpy as np
import threading
from enum import Enum, auto
from utils import emit
from speech_to_text.asrInterface import (
    IVADModel,
    IOnlineTranscriptionModel,
    IOfflineTranscriptionModel,
    IUnifiedTranscriptionModel,
    IVerificationModel,
)


class SpeechRecognizerState(Enum):
    ONLINE = auto()
    OFFLINE = auto()
    IDLE = auto()


class SpeechRecognizer:
    def __init__(
        self,
        vad_model: IVADModel,
        online_model: IOnlineTranscriptionModel = None,
        offline_model: IOfflineTranscriptionModel = None,
        unified_model: IUnifiedTranscriptionModel = None,
        speaker_verifier: IVerificationModel = None,
        socketio=None,
        RATE=16000,
        CHUNK=9600,
        accumulated_speech_threshold=50,
        is_ending_counter_threshold=2,
        is_idle_counter_threshold=8,
    ):
        self.vad_model = vad_model

        if unified_model is not None:
            self.online_model = self.offline_model = unified_model
        else:
            self.online_model = online_model
            self.offline_model = offline_model

        if self.online_model is None or self.offline_model is None:
            raise ValueError("Both online and offline models must be provided.")

        self.RATE = RATE
        self.CHUNK = CHUNK

        self.listening_to_user_event = threading.Event()
        self.state = SpeechRecognizerState.IDLE

        self.is_ending_counter = 0
        self.is_ending_counter_threshold = is_ending_counter_threshold

        self.is_idle_counter = 0
        self.is_idle_counter_threshold = is_idle_counter_threshold

        self.text_2pass_offline = ""
        self.text_2pass_online = ""

        self.speaker_verifier = speaker_verifier
        self.initial_speaker = None

        self.accumulated_speech = []
        self.accumulated_speech_threshold = accumulated_speech_threshold

        self.socketio = socketio

    def process_audio_chunk(self, audio_data):
        """Process a single audio chunk using VAD and ASR models."""
        if audio_data is None:
            return
        emit(self.socketio, "user_idle_counter", {"counter": self.is_idle_counter_threshold - self.is_idle_counter})
        user_speaking = self.vad_model.detect(audio_data) and self.speaker_verifier.verify(audio_data)

        # Online
        if user_speaking:
            online_transcription = self.online_model.online_transcribe(audio_data)

            self.text_2pass_online += online_transcription
            self.accumulated_speech.append(audio_data)
            emit(self.socketio, "online_transcription", {"message": online_transcription})

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
                audio_data = np.concatenate(self.accumulated_speech)

                self.speaker_verifier.set_initial_reference(audio_data)

                offline_transcription = self.offline_model.offline_transcribe(audio_data)
                self.text_2pass_online = ""
                self.text_2pass_offline += offline_transcription
                emit(self.socketio, "offline_transcription", {"message": offline_transcription})

                self.set_state(SpeechRecognizerState.OFFLINE)

        # Idle
        if not self.state == SpeechRecognizerState.IDLE and not user_speaking:
            self.is_idle_counter += 1
            if self.is_idle_counter >= self.is_idle_counter_threshold:
                self.set_state(SpeechRecognizerState.IDLE)

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

    def reset_flags(self):
        if self.state == SpeechRecognizerState.IDLE:
            self.is_idle_counter = 0
        elif self.state == SpeechRecognizerState.OFFLINE:
            self.accumulated_speech = []
            self.is_ending_counter = 0
            self.is_idle_counter = 0
            self.online_model.reset_online_cache()
            self.update_display()
        elif self.state == SpeechRecognizerState.ONLINE:
            self.is_ending_counter = 0
            self.is_idle_counter = 0
            self.update_display()
