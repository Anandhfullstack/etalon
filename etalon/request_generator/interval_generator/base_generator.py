from abc import ABC, abstractmethod

from etalon.config.config import BaseRequestIntervalGeneratorConfig


class BaseRequestIntervalGenerator(ABC):
    def __init__(self, config: BaseRequestIntervalGeneratorConfig):
        self.config = config

    @abstractmethod
    def get_next_inter_request_time(self) -> float:
        pass
