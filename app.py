from flask import Flask, render_template
from flask_socketio import SocketIO
from utils import emit
from assistant import PersonalAssistant
import numpy as np

# Initialize Flask app and Flask-SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# Initialize the personal assistant, passing the emit callback
assistant = PersonalAssistant(socketio=socketio)


# import threading
# import time
# import random
# import string


# class MockPersonalAssistant:
#     def __init__(self, socketio):
#         self.socketio = socketio
#         self.audio_queue = []

#     def generate_random_string(self, length=10):
#         # Generates a random string of a given length
#         return "".join(random.choices(string.ascii_letters + string.digits, k=length))

#     def run(self):
#         def process_audio():
#             counter = 0
#             while True:
#                 counter += 1
#                 # Simulating audio processing delay
#                 time.sleep(1)

#                 # Emit a mock response with random string for online transcription
#                 self.socketio.emit(
#                     "online_transcription",
#                     {"message": f"{counter} online"},
#                 )

#                 time.sleep(2)

#                 # Emit a mock response with random string for offline transcription
#                 self.socketio.emit(
#                     "offline_transcription",
#                     {"message": f"{counter} offline"},
#                 )

#                 time.sleep(1)

#                 # Emit that the LLM is starting
#                 self.socketio.emit("llm_started", {"started": True})

#                 time.sleep(3)

#                 # Emit a mock LLM answer with a random string

#                 self.socketio.emit("llm_answer", {"message": f"{counter} llm"})

#                 time.sleep(1)

#                 self.socketio.emit("llm_stopped", {"stopped": True})

#         return threading.Thread(target=process_audio)


# # # Initialize the mock assistant
# assistant = MockPersonalAssistant(socketio=socketio)

# Start the mock audio processing thread
process_audio_thread = assistant.run()
process_audio_thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("audio_data")
def handle_audio_data(mic_bytes):
    print(f"Received audio data: {len(mic_bytes)} bytes")

    audio_array = np.frombuffer(mic_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    assistant.audio_queue.put(audio_array)

    socketio.emit("audio_ack", {"message": "Audio received and processed"})


@socketio.on("connect")
def handle_connect():
    emit(
        socketio,
        "user_idle_counter_threshold",
        {"threshold": assistant.speech_recognizer.is_idle_counter_threshold},
    )


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8080)
    process_audio_thread.join()
