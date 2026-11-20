from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__():
        pass

    @abstractmethod
    def params() -> list:
        pass

    