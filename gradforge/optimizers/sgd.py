from ..core.tensor import Tensor
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001) -> None:
        super().__init__(learning_rate)

    def step(self, params: list[Tensor]):
        for param in params:
            if param.data.shape != param.grad.shape:
                param.sum_grads()
            param.data -= param.grad * self.learning_rate

    def zero_grad(self, params: list[Tensor]):
        for param in params:
            param.grad = 0  