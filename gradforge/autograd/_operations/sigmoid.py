from math import e

from ...autograd.operation import Operation


class Sigmoidal(Operation):
    def __init__(self) -> None:
        super().__init__(None)
        self.result = 0

    def forward(self, x):

        result = 1 / (1 + e**(-x))
        self.result = result

        return result 

    
    def backward(self, result_grad) -> list:
        
        x_grad = result_grad * self.result * (1 - self.result)

        return [x_grad]

        