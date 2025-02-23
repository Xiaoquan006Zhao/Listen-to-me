from asr import SpeechRecognizer, SpeechRecognizerState
from tts import SpeechGenerator
import queue
import threading
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import time
import numpy as np
import pyaudio


class PersonalAssistant:
    def __init__(self):
        """Initialize the personal assistant with ASR, TTS, and LLM systems."""
        self.speech_recognizer = SpeechRecognizer()
        self.speech_generator = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
        self.audio_queue = queue.Queue()
        self.llm = ChatOllama(model="wizardlm2:7b")
        self.print_lock = threading.Lock()
        self.chat_history = []
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

        # Add user input to chat history with instructions
        self.chat_history.append(
            HumanMessage(
                content=transcription + " Only answer in English and do not use Markdown. Use normal punctuation."
            )
        )

        # Prepare messages for LLM (include chat history)
        messages = [*self.chat_history]

        def llm_thread():
            """Handle streaming LLM responses and generate speech."""
            speech_buffer = ""
            response_content = ""

            for chunk in self.llm.stream(messages):
                if self.speech_recognizer.state != SpeechRecognizerState.IDLE:
                    self.speech_generator.interrupt()
                    break

                speech_buffer += chunk.content
                response_content += chunk.content

                # Print the chunk immediately
                with self.print_lock:
                    print(chunk.content, end="")

                # Buffer until a sentence is complete for natural speech generation
                if len(speech_buffer) > 50 and any(speech_buffer.endswith(p) for p in (".", "!", "?", "\n")):
                    self.speech_generator.add_text_to_queue(speech_buffer)
                    speech_buffer = ""

            # Process any remaining content in the buffer
            if speech_buffer and not self.speech_generator.stop_event.is_set():
                self.speech_generator.add_text_to_queue(speech_buffer)

            print()
            if response_content:
                self.chat_history.append(AIMessage(content=response_content))

            print("------" * 10)

        def llm_finished_speech_ongoing_thread(previous_thread):
            """Wait for the LLM thread to finish and handle ongoing speech."""
            previous_thread.join()

            while (
                not self.speech_generator.text_queue.empty()
                or not self.speech_generator.audio_queue.empty()
                or self.speech_generator.audio_player.is_audio_playing()
                or self.speech_generator.is_kokoro_running
            ):
                if self.speech_recognizer.state != SpeechRecognizerState.IDLE:
                    self.speech_generator.interrupt()
                    return
                time.sleep(0.1)

            self.speech_generator.interrupt()
            print("*******" * 10)

        # Start LLM and speech monitoring threads
        llm_thread_instance = threading.Thread(target=llm_thread)
        llm_thread_instance.start()
        threading.Thread(target=llm_finished_speech_ongoing_thread, args=(llm_thread_instance,)).start()

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
