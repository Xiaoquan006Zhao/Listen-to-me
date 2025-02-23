from asr import SpeechRecognizer, SpeechRecognizerState
from tts import SpeechGenerator
import queue
import threading
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class PersonalAssistant:
    def __init__(self):
        """Initialize the personal assistant with ASR and LLM systems."""
        self.speechRecognizer = SpeechRecognizer()
        self.speechGenerator = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
        self.audio_queue = queue.Queue()
        self.llm = ChatOllama(model="wizardlm2:7b")
        self.print_lock = threading.Lock()
        self.chat_history = []  # List to store chat history

    def process_audio(self):
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                break

            self.speechRecognizer.process_audio_chunk(audio_data)

            transcription = self.speechRecognizer.get_transcription().strip()

            if self.speechRecognizer.state == SpeechRecognizerState.IDLE and transcription:
                print("Processing with LLM...")
                self.speechRecognizer.reset_external_transcription()
                self.process_with_llm(transcription)

    def process_with_llm(self, transcription):
        # Add user message to chat history
        self.chat_history.append(HumanMessage(content=transcription + "The answe should only be in English!"))

        # Prepare messages for LLM (include system message and chat history)
        messages = [
            # SystemMessage(content="Provide anwers only in English."),
            *self.chat_history,  # Include entire chat history
        ]

        # threading is necessary for state upate to interupt by user speaking
        def llm_thread():
            # Buffer to accumulate chunks for more natural speech. Token level speech genration is not good.
            speechBuffer = ""
            response_content = ""
            for chunk in self.llm.stream(messages):
                if self.speechRecognizer.state != SpeechRecognizerState.IDLE and self.speechRecognizer.speaker_verified:
                    break
                speechBuffer += chunk.content

                # Immediately print the chunk
                with self.print_lock:
                    print(chunk.content, end="")
                    response_content += chunk.content

                # Buffer until a sentence is complete to generate speech
                if any(speechBuffer.endswith(p) for p in (".", "!", "?", "\n", "。")):
                    print(speechBuffer)
                    self.speechGenerator.stream_start(speechBuffer)
                    speechBuffer = ""

            # Process any remaining content in the buffer
            if speechBuffer:
                self.speechGenerator.stream_start(speechBuffer)

            print()
            # Add assistant's response to chat history
            if response_content:
                self.chat_history.append(AIMessage(content=response_content))

        threading.Thread(target=llm_thread).start()

    def record_audio(self, RATE=16000, CHUNK=9600):
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
                self.audio_queue.put(audio_array)

        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            stream.stop_stream()
            stream.close()


def main():
    assistant = PersonalAssistant()

    record_thread = threading.Thread(target=assistant.record_audio)
    record_thread.start()

    try:
        assistant.process_audio()
    except KeyboardInterrupt:
        print("Processing stopped.")
    finally:
        record_thread.join()


if __name__ == "__main__":
    main()
