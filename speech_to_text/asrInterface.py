from abc import ABC, abstractmethod


class IVADModel(ABC):
    @abstractmethod
    def detect(self, audio_data):
        """
        Process the audio data and return True if speech is detected, False otherwise.
        """
        pass


class IOnlineTranscriptionModel(ABC):
    @abstractmethod
    def transcribe(self, audio_data, **kwargs):
        pass

    def set_language(self, language):
        pass

    def reset_online_cache(self):
        pass


class IOfflineTranscriptionModel(ABC):
    @abstractmethod
    def transcribe(self, audio_data):
        pass


class IUnifiedTranscriptionModel(ABC):
    @abstractmethod
    def online_transcribe(self, audio_data):
        pass

    @abstractmethod
    def offline_transcribe(self, audio_data):
        pass

    def set_language(self, language):
        pass

    def reset_online_cache(self):
        pass


class IVerificationModel(ABC):
    @abstractmethod
    def verify(self, audio_data, reference_audio, threshold):
        pass

    def set_initial_reference(self, reference_audio):
        pass
