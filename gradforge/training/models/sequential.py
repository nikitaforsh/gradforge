from ..layers.layer import Layer
from ...core.tensor import Tensor
class Sequential:
    def __init__(self, *layers: list[Layer]):
        self.layers: list[Layer] = layers
    
    @property
    def params(self):
        for layer in self.layers:
            yield layer.params