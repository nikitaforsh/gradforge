from ..layers.layer import Layer


class Sequential:
    def __init__(self, *layers: list[Layer]):
        self.layers: list[Layer] = layers
    
    # TODO: Create learning cycle
    @property
    def params(self):
        for layer in self.layers:
            yield layer.params