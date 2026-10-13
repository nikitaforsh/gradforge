from gradforge.core.tensor import Tensor
from gradforge.backend.backend_manager import BackendManager
from gradforge.autograd.standard_engine import StandardEngine
from gradforge.optimizers.sgd import SGD

class linearModel:
    def __init__(self):
        self.backend = BackendManager().get_backend()
        self.engine = StandardEngine()
        self.optimizer = SGD(learning_rate=0.001)

        self.x = Tensor([[1, 2, 3, 4, 5]])
        self.y = Tensor([[2, 4, 6, 8, 10]])

        self.w = Tensor(self.backend.array_zeros((self.x.data.shape[1], self.y.data.shape[1])), requires_grad=True)
        self.b = Tensor([0], grad_sum_axes=(1, 1), requires_grad=True)


    def learn(self, epochs):
        for epoch in range(epochs):
            predict = self.w * self.x + self.b
            loss = self.backend.mean((predict - self.y)**2)

            self.optimizer.zero_grad([self.w, self.b])
            self.engine.backward(loss)
            self.optimizer.step([self.w, self.b])

            if epoch % 100 == 0:
                print(f'Weight: {self.w}  ||  Bias: {self.b} ||')

model = linearModel()
model.learn(10000)
print(model)