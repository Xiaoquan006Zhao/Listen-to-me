from funasr import AutoModel
from modelscope.pipelines import pipeline
from utils import postprocess_funasr_result, emit, extract_language_code
from speech_to_text.asrInterface import (
    IVADModel,
    IOnlineTranscriptionModel,
    IOfflineTranscriptionModel,
    IUnifiedTranscriptionModel,
    IVerificationModel,
)


class FunASRVAD(IVADModel):
    def __init__(self, model_name, verbose=False):
        disable = not verbose
        self.model = AutoModel(model=model_name, disable_log=disable, disable_pbar=disable, disable_update=disable)

    def detect(self, audio_data):
        # Return True if speech is detected, False otherwise
        vad_res = self.model.generate(input=audio_data)
        return len(vad_res[0]["value"]) > 0


class FunASRUnifiedTranscription(IUnifiedTranscriptionModel):
    def __init__(self, model_name, verbose=False):
        disable = not verbose
        self.model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            hub="hf",
            disable_log=disable,
            disable_pbar=disable,
            disable_update=disable,
        )

        self.language = "auto"
        self.online_cache = {}

    def online_transcribe(self, audio_data):
        online_res = self.model.generate(
            input=audio_data,
            cache=self.online_cache,
            language=self.language if self.language is not None else "auto",
            use_itn=True,
            batch_size_s=60,
        )

        return postprocess_funasr_result(online_res, remove_punctuation=True)

    def offline_transcribe(self, audio_data):
        offline_res = self.model.generate(
            input=audio_data,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )

        self.language = extract_language_code(offline_res[0]["text"])

        return postprocess_funasr_result(offline_res)

    def set_language(self, language):
        self.language = language

    def reset_online_cache(self):
        self.online_cache = {}


class FunASRSpeakerVerification(IVerificationModel):
    def __init__(self, model_name, threshold=0.35):
        self.pipeline = pipeline(task="speaker-verification", model=model_name)
        self.reference_audio = None
        self.threshold = threshold

    def verify(self, audio_data):
        if self.reference_audio is None:
            return True

        result = self.pipeline([audio_data, self.reference_audio], thr=self.threshold)
        return result.get("text", "").lower() == "yes"

    def set_initial_reference(self, reference_audio):
        if self.reference_audio is None:
            self.reference_audio = reference_audio
