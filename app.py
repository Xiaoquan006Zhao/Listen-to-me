from flask import Flask, render_template
from flask_socketio import SocketIO
from utils import emit
from voiceAssistant import VoiceAssistant
import numpy as np
from speech_to_text.funASR import (
    FunASRVAD,
    FunASRUnifiedTranscription,
    FunASRSpeakerVerification,
)

from speech_to_text.speechRecognizer import SpeechRecognizer, SpeechRecognizerState
from text_to_speech.speechGenerator import SpeechGenerator
from llm import AnswerGenerator
from text_to_speech.kokoroModel import kokoroModel

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# speech to text
vad_model = FunASRVAD(model_name="fsmn-vad")
unified_model = FunASRUnifiedTranscription(model_name="FunAudioLLM/SenseVoiceSmall")
sv_model = FunASRSpeakerVerification(model_name="iic/speech_campplus_sv_zh-cn_16k-common")
speech_recognizer = SpeechRecognizer(
    vad_model=vad_model, unified_model=unified_model, sv_model=sv_model, socketio=socketio
)
interrupt_event = speech_recognizer.listening_to_user_event

# text to speech
kokoro = kokoroModel("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
speech_generator = SpeechGenerator(model=kokoro, interrupt_event=interrupt_event, socketio=socketio)

# llm
answer_generator = AnswerGenerator(model="wizardlm2:7b", socketio=socketio, interrupt_event=interrupt_event)

# assistant
assistant = VoiceAssistant(
    speech_recognizer,
    speech_generator,
    answer_generator,
    socketio,
)
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
