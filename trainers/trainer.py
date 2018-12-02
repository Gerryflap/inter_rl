from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def add_experiences(self, experiences):
        pass

    @abstractmethod
    def params_to_json(self):
        pass

    @abstractmethod
    def train_loop(self):
        pass