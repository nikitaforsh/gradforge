from gradforge.core.tensor import Tensor
from gradforge.backend.backend_manager import BackendManager
from gradforge.autograd.standard_engine import StandardEngine
from gradforge.optimizers.sgd import SGD

class Linear:
    def __init__(self):
        self.backend   = BackendManager().get_backend()
        self.engine    = StandardEngine()
        self.optimizer = SGD(learning_rate=1e-3)

        self.x = Tensor([[1], [2], [3], [4], [5],  [6],  [7],  [8],  [9],  [10]])
        self.y = Tensor([[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]])

        self.w = Tensor(self.backend.array_zeros((self.x.data.shape[1], self.y.data.shape[1])), requires_grad=True)
        self.b   = Tensor([0], requires_grad=True)

        def learning(self, epochs):
            for epoch in range(epochs):
                prediction = self.w + self.x + self.b
                loss = self.backend.mean((prediction - self.y)**2)

                self.optimizer.zero_grad([self.w, self.b])
                self.engine.backward(loss)
                self.optimizer.step()

                if (epoch + 1) % 100 == 0:
                    print(f'||  Weight:  {self.w}  ||'
                          f'  Bias:  {self.b}  ||')

model = Linear()
model.learning(10000)