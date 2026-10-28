from ...autograd.operation import Operation


class BinaryCrossentropy(Operation):
    def __init__(self, backend, epsilon=1e-5) -> None:
        super().__init__(backend)
        self.epsilon = 1e-5
        self.p = None
        self.y = None

    def forward(self, p, y):
        self.p = self._backend.clip(p, self.epsilon, 1 - self.epsilon)
        loss = - (self.y * self._backend.log(self.p) + (1 - self.y) * self._backend.log(1 - self.p))
        return self._backend.mean(loss)

    
    def backward(self, result_grad) -> tuple:
        n = self.y.shape
        grad = ((self.p - self.y) / (self.p * (1 - self.p))) / n

        return [grad]
        