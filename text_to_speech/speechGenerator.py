import asyncio
import threading
import base64
from utils import preprocess_before_generation, emit
from text_to_speech.ttsInterface import ITTSModel


class SpeechGenerator:
    def __init__(
        self,
        model: ITTSModel,
        interrupt_event,
        socketio=None,
        buffer_threshold=50,
    ):
        self.model = model

        self.socketio = socketio
        self.buffer_threshold = buffer_threshold

        self.text_buffer = ""
        self.text_queue = asyncio.Queue()
        self.interrupt_event = interrupt_event

        self.loop = None
        self.loop_ready = threading.Event()  # New event to signal that the loop is ready
        self.last_task_event = asyncio.Event()

        # Start the asyncio loop in a dedicated thread.
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop_ready.set()  # Signal that the loop is ready
            self.loop.run_forever()

        threading.Thread(target=run_loop, daemon=True).start()

    async def generate_speech(self, text):
        text = preprocess_before_generation(text)

        stream = self.model.synthesize(text)

        async for samples, sample_rate in stream:
            if self.interrupt_event.is_set():
                break
            # Convert samples to int16 PCM format.
            samples = (samples * 32767).astype("int16")
            audio_base64 = base64.b64encode(samples.tobytes()).decode("utf-8")
            emit(self.socketio, "audio_stream", {"samplerate": sample_rate, "samples": audio_base64, "stopped": False})

    async def process_queue(self):
        while not self.interrupt_event.is_set():
            text = await self.text_queue.get()
            await self.generate_speech(text)
            self.text_queue.task_done()

            if self.last_task_event.is_set() and self.text_queue.empty():
                emit(self.socketio, "all_speech_sent", {"all_sent": True})
                break

        self.stop()

    def add_text(self, text, buffered=True):
        # Wait until the event loop is ready.
        self.loop_ready.wait()
        self.text_buffer += text

        if buffered:
            self.last_task_event.set()

        if self.text_buffer and (
            not buffered or (len(self.text_buffer) >= self.buffer_threshold and self.text_buffer[-1] in " ,:;.!?\n")
        ):
            asyncio.run_coroutine_threadsafe(self.text_queue.put(self.text_buffer), self.loop)
            self.text_buffer = ""

    def stop(self):
        # Use get_nowait() to clear the queue without awaiting.
        try:
            while True:
                self.text_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        self.text_buffer = ""

    def start(self):
        self.last_task_event.clear()
        asyncio.run_coroutine_threadsafe(self.process_queue(), self.loop)
