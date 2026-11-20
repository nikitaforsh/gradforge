from abc import ABC, abstractmethod


class SplitCriteria(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_output(self, y_left, y_right):
        pass