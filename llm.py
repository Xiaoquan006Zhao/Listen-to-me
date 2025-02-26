import queue
import threading
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from utils import emit


class AnswerGenerator:
    def __init__(self, model, socketio):
        self.llm = ChatOllama(model=model)
        self.chat_history = []
        self.interrupt_event = None
        self.socketio = socketio

    def set_interrupt_event(self, interrupt_event):
        self.interrupt_event = interrupt_event

    def stream_answer(self, current_user_message):
        self.chat_history.append(
            HumanMessage(
                content=current_user_message
                + " Only answer in English and do not use Markdown. Use normal punctuation."
            )
        )
        messages = [*self.chat_history]
        response_buffer = ""

        for chunk in self.llm.stream(messages):
            if self.interrupt_event.is_set():
                break
            llm_generated_text = chunk.content
            yield llm_generated_text
            emit(self.socketio, "llm_answer", {"message": llm_generated_text})
            response_buffer += llm_generated_text

        if response_buffer:
            self.chat_history.append(AIMessage(content=response_buffer))

        emit(self.socketio, "llm_stopped", {"stopped": True})


if __name__ == "__main__":
    import threading
    import time

    def run_generator(stop_event):
        answer_generator = AnswerGenerator()

        answer_generator.set_interrupt_event(stop_event)

        # Signal the generator to stop
        for chunk in answer_generator.stream_answer("Give me a list of 100 fruits."):
            print(chunk)

    stop_event = threading.Event()
    generator_thread = threading.Thread(target=run_generator, args=(stop_event,))
    generator_thread.start()

    time.sleep(5)
    print("Stopping generator...")
    stop_event.set()
