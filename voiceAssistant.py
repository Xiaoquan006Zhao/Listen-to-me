from speech_to_text.speechRecognizer import SpeechRecognizer, SpeechRecognizerState
from text_to_speech.speechGenerator import SpeechGenerator
from llm import AnswerGenerator
import queue
import threading
from utils import emit


class VoiceAssistant:
    def __init__(
        self,
        speech_recognizer: SpeechRecognizer,
        speech_generator: SpeechGenerator,
        answer_generator: AnswerGenerator,
        socketio=None,
    ):
        """Initialize the personal assistant with ASR, TTS, and LLM systems."""
        self.speech_recognizer = speech_recognizer
        self.speech_generator = speech_generator
        self.answer_generator = answer_generator
        self.socketio = socketio

        self.audio_queue = queue.Queue()
        self.threads = []

    def process_audio(self):
        """Process audio chunks from the queue and handle transcription."""
        while True:
            # Since queue.get() is a blocking operation, the while loop is not busy-waiting
            audio_data = self.audio_queue.get()

            self.speech_recognizer.process_audio_chunk(audio_data)
            transcription = self.speech_recognizer.get_transcription().strip()

            if transcription and self.speech_recognizer.state == SpeechRecognizerState.IDLE:
                print("Processing with LLM...")
                emit(self.socketio, "llm_started", {"started": True})
                self.speech_recognizer.reset_external_transcription()
                self.process_with_llm(transcription)

    def process_with_llm(self, transcription):
        """Process the transcription with the LLM and generate a response."""
        for thread in threading.enumerate():
            print(thread)

        # Start the speech generation thread
        # Allow playback at client side
        emit(self.socketio, "listening_to_user", {"listening": False})
        self.speech_generator.start()

        def llm_thread():
            print("LLM thread started")
            for token in self.answer_generator.stream_answer(transcription):
                self.speech_generator.add_text(token)

            # Process any remaining content in the buffer
            if not self.speech_generator.interrupt_event.is_set():
                self.speech_generator.add_text("", buffered=False)

        llm_thread_instance = threading.Thread(target=llm_thread)
        llm_thread_instance.start()

    def run(self):
        process_audio_thread = threading.Thread(target=self.process_audio)
        self.threads.append(process_audio_thread)
        return process_audio_thread
