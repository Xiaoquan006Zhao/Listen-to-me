from asr import SpeechRecognizer, SpeechRecognizerState
from tts import SpeechGenerator
from llm import AnswerGenerator
import queue
import threading
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
import pyaudio


class PersonalAssistant:
    def __init__(self):
        """Initialize the personal assistant with ASR, TTS, and LLM systems."""
        self.speech_recognizer = SpeechRecognizer()
        self.speech_generator = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
        self.audio_queue = queue.Queue()

        self.answer_generator = AnswerGenerator(model="wizardlm2:7b")
        self.answer_generator.set_interrupt_event(self.speech_recognizer.listening_to_user_event)

        self.print_lock = threading.Lock()
        self.threads = []

    def process_audio(self):
        """Process audio chunks from the queue and handle transcription."""
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:  # Sentinel value to stop processing
                break

            self.speech_recognizer.process_audio_chunk(audio_data)
            transcription = self.speech_recognizer.get_transcription().strip()

            if self.speech_recognizer.state == SpeechRecognizerState.IDLE and transcription:
                print("Processing with LLM...")
                self.speech_recognizer.reset_external_transcription()
                self.process_with_llm(transcription)

    def process_with_llm(self, transcription):
        """Process the transcription with the LLM and generate a response."""
        print("------")
        print("Thread count:", threading.active_count())
        print("------")

        # Restart the speech generation thread
        speech_thread = self.speech_generator.restart()
        self.threads.append(speech_thread)
        speech_thread.start()

        def llm_thread():
            for token in self.answer_generator.stream_answer(transcription):
                with self.print_lock:
                    print(token, end="")
                self.speech_generator.add_text_to_queue(token)

            # Process any remaining content in the buffer
            if not self.speech_generator.stop_event.is_set():
                self.speech_generator.add_text_to_queue("", buffered=False)

            print()
            print("------" * 10)

        def listening_for_user_interrupt_thread(previous_llm_thread):
            previous_llm_thread.join()
            self.speech_recognizer.listening_to_user_event.wait()
            print("User interrupting...")
            self.speech_generator.interrupt()
            print("*****")

        def speech_natural_completion_thread(previous_llm_thread):
            previous_llm_thread.join()
            self.speech_generator.task_completed_event.wait()
            print("Natural completion...")
            # Not logically ideal, as it's set to stop the listening thread, not to indicate actual user speaking
            self.speech_recognizer.listening_to_user_event.set()
            print("*****")

        # Start LLM and speech monitoring threads
        llm_thread_instance = threading.Thread(target=llm_thread)
        llm_thread_instance.start()
        threading.Thread(target=listening_for_user_interrupt_thread, args=(llm_thread_instance,)).start()
        threading.Thread(target=speech_natural_completion_thread, args=(llm_thread_instance,)).start()

    def record_audio(self, rate=16000, chunk_size=9600):
        """Record audio from the microphone and put chunks into the queue."""
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk_size)
        print("Recording...")

        try:
            while True:
                audio_data = stream.read(chunk_size)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / 32768.0
                self.audio_queue.put(audio_array)

        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


def main():
    """Main function to initialize and run the personal assistant."""
    assistant = PersonalAssistant()

    record_thread = threading.Thread(target=assistant.record_audio)
    assistant.threads.append(record_thread)
    record_thread.start()

    try:
        assistant.process_audio()
    except KeyboardInterrupt:
        print("Processing stopped.")
    finally:
        for thread in assistant.threads:
            thread.join()


if __name__ == "__main__":
    main()
