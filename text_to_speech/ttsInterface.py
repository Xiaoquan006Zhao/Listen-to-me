from abc import ABC, abstractmethod


class ITTSModel(ABC):
    @abstractmethod
    def synthesize(self, text):
        # Stream audio creation asynchronously in the background, yielding chunks as they are processed.
        pass
