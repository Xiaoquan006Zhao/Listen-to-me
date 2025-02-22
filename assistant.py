from asr import SpeechRecognizer, State
from utils import record_audio
import queue
import threading
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


class PersonalAssistant:
    def __init__(self):
        """Initialize the personal assistant with ASR and LLM systems."""
        self.speechRecognizer = SpeechRecognizer()
        self.audio_queue = queue.Queue()
        self.llm = ChatOllama(model="wizardlm2:7b")

    def process_audio(self):
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                break

            self.speechRecognizer.process_audio_chunk(audio_data)

            transcription = self.speechRecognizer.get_transcription().strip()

            print(transcription)
            if self.speechRecognizer.state == State.IDLE and transcription:
                print("Processing with LLM...")
                self.speechRecognizer.reset_external_transcription()
                self.process_with_llm(transcription)

    def process_with_llm(self, transcription):
        messages = [
            SystemMessage(content="Answer concisely unless specified."),
            HumanMessage(content=f"{transcription}"),
        ]

        for chunk in self.llm.stream(messages):
            print(chunk.content, end="")


def main():
    assistant = PersonalAssistant()

    record_thread = threading.Thread(target=record_audio, args=(assistant.audio_queue,))
    record_thread.start()

    try:
        assistant.process_audio()
    except KeyboardInterrupt:
        print("Processing stopped.")
    finally:
        record_thread.join()


if __name__ == "__main__":
    main()
