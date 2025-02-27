from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from utils import emit


class AnswerGenerator:
    def __init__(
        self,
        model,
        interrupt_event,
        socketio=None,
    ):
        self.llm = ChatOllama(model=model)
        self.chat_history = []
        self.interrupt_event = interrupt_event
        self.socketio = socketio

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
