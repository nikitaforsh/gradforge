from abc import ABC, abstractmethod

from ..backend.backend_manager import BackendManager


class Engine(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._backend = BackendManager.get_backend()

    @abstractmethod
    def backward():
        pass