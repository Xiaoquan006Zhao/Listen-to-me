from asr import SpeechRecognizer, SpeechRecognizerState
from tts import SpeechGenerator
import queue
import threading
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import time


# TODO Avoid endless spin loops, use events to coordinate threads
class PersonalAssistant:
    def __init__(self):
        """Initialize the personal assistant with ASR and LLM systems."""
        self.speechRecognizer = SpeechRecognizer()
        self.speechGenerator = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
        self.audio_queue = queue.Queue()
        self.llm = ChatOllama(model="wizardlm2:7b")
        self.print_lock = threading.Lock()
        self.chat_history = []
        self.threads = []

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
        print("------")
        print("thread count: ", threading.active_count())
        print("------")

        speech_thread = self.speechGenerator.restart()
        self.threads.append(speech_thread)
        speech_thread.start()

        self.chat_history.append(
            HumanMessage(
                content=transcription
                + "Only Answer with English and no other language should be used! And do not use any Markdown. Just use normal English punctuation."
            )
        )

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
                # Although the SpeechRecognizerState should already verify the speaker, we check again here
                if self.speechRecognizer.state != SpeechRecognizerState.IDLE and self.speechRecognizer.speaker_verified:
                    self.speechGenerator.interrupt()
                    break
                speechBuffer += chunk.content

                # Immediately print the chunk
                with self.print_lock:
                    print(chunk.content, end="")
                    response_content += chunk.content

                # Buffer until a sentence is complete to generate speech
                if len(speechBuffer) > 50 and any(speechBuffer.endswith(p) for p in (".", "!", "?", "\n", "。")):
                    self.speechGenerator.add_text_to_queue(speechBuffer)
                    speechBuffer = ""

            # Process any remaining content in the buffer
            if speechBuffer and not self.speechGenerator.stop_event.is_set():
                self.speechGenerator.add_text_to_queue(speechBuffer)

            print()
            if response_content:
                self.chat_history.append(AIMessage(content=response_content))

            print("------" * 10)

        def llm_finished_speech_ongoing_thread(previous_thread):
            previous_thread.join()
            while (
                not self.speechGenerator.text_queue.empty()
                or not self.speechGenerator.audio_queue.empty()
                or self.speechGenerator.audio_player.is_audio_playing()
                or self.speechGenerator.is_kokoro_running
            ):
                if self.speechRecognizer.state != SpeechRecognizerState.IDLE and self.speechRecognizer.speaker_verified:
                    self.speechGenerator.interrupt()
                    return
                time.sleep(0.1)

            self.speechGenerator.interrupt()
            print("*******" * 10)

        thread = threading.Thread(target=llm_thread)
        thread.start()
        threading.Thread(target=llm_finished_speech_ongoing_thread, args=(thread,)).start()

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
    assistant.threads.append(record_thread)

    # speech_thread = threading.Thread(target=assistant.speechGenerator.run)
    # assistant.threads.append(speech_thread)

    for thread in assistant.threads:
        thread.start()
    try:
        assistant.process_audio()
    except KeyboardInterrupt:
        print("Processing stopped.")
    finally:
        for thread in assistant.threads:
            thread.join()


if __name__ == "__main__":
    main()
