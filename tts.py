import sounddevice as sd
from kokoro_onnx import Kokoro
import asyncio
import threading


class SpeechGenerator:
    def __init__(self, model_path, voices_path, voice="am_echo", speed=1.0, lang="en-us"):
        self.kokoro = Kokoro(model_path, voices_path)
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.interrupted = False  # Flag to stop streaming
        self.thread = None

    async def stream_text(self, text):
        """Blocking function to stream speech"""
        stream = self.kokoro.create_stream(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )

        async for samples, sample_rate in stream:
            sd.play(samples, sample_rate)
            sd.wait()

    def stream_start(self, text):
        asyncio.run(self.stream_text(text))

        # """Start streaming in a separate thread"""
        # self.interrupted = False  # Reset interruption flag
        # self.thread = threading.Thread(target=self.stream_text, args=(text,))
        # self.thread.start()


if __name__ == "__main__":
    text = """
    We've just been hearing from Matthew Cappucci, a senior meteorologist at the weather app MyRadar, who says Kansas City is seeing its heaviest snow in 32 years - with more than a foot (30 to 40cm) having come down so far.

    Despite it looking as though the storm is slowly moving eastwards, Cappucci says the situation in Kansas and Missouri remains serious.

    He says some areas near the Ohio River are like "skating rinks", telling our colleagues on Newsday that in Missouri in particular there is concern about how many people have lost power, and will lose power, creating enough ice to pull power lines down.

    Temperatures are set to drop in the next several days, in may cases dipping maybe below minus 10 to minus 15 degrees Celsius for an extended period of time.

    There is a special alert for Kansas, urging people not to leave their homes: "The ploughs are getting stuck, the police are getting stuck, everybody’s getting stuck - stay home."
    """

    tts = SpeechGenerator("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")
    tts.stream_start(text)
    import time

    time.sleep(2)  # Let it play for 2 seconds

    tts.stream_stop()  # Stop the speech
