from flask import Flask, render_template
from flask_socketio import SocketIO
from utils import emit
from assistant import PersonalAssistant
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

assistant = PersonalAssistant(socketio=socketio)
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
