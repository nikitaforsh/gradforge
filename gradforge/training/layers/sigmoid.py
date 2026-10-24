from ...autograd._operations.sigmoid import Sigmoidal
from ...core.tensor import Tensor
from .layer import Layer


class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__() 
        self.sigmoid = Sigmoidal()

        
    def predict(self, x: Tensor) -> Tensor:
        pred = Tensor(self.sigmoid.forward(x.data), requires_grad=x.requires_grad)
        pred.grad_fn = [[x], self.sigmoid]
        return pred

    @property
    def params(self) -> list[Tensor]:
        return []