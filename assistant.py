from asr import SpeechRecognizer, SpeechRecognizerState
from tts import SpeechGenerator
from llm import AnswerGenerator
import queue
import threading
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
from utils import emit, record_audio


class PersonalAssistant:
    def __init__(self, socketio=None):
        """Initialize the personal assistant with ASR, TTS, and LLM systems."""
        self.speech_recognizer = SpeechRecognizer(socketio=socketio)
        interrupt_event = self.speech_recognizer.listening_to_user_event

        self.speech_generator = SpeechGenerator(
            "kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin", interrupt_event=interrupt_event, socketio=socketio
        )
        self.answer_generator = AnswerGenerator(
            model="wizardlm2:7b", socketio=socketio, interrupt_event=interrupt_event
        )

        self.audio_queue = queue.Queue()
        self.socketio = socketio
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


if __name__ == "__main__":
    assistant = PersonalAssistant()
    process_audio_thread = assistant.run()
    process_audio_thread.start()

    threading.Thread(target=record_audio, args=(assistant.audio_queue,)).start()

    process_audio_thread.join()
