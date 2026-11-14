from gradforge.core.tensor import Tensor
from gradforge.backend.backend_manager import BackendManager
from gradforge.autograd.standard_engine import StandardEngine
from gradforge.optimizers.sgd import SGD
from gradforge.autograd._operations.sigmoidal import Sigmoidal
from gradforge.autograd._operations.loss.bce import BinaryCrossentropy

class Classification():
    def __init__(self):
        self.optimizer = SGD(learning_rate=1e-3)
        self.bacnkend  = BackendManager().get_backend()
        self.engine    = StandardEngine()
        self.sigmoid   = Sigmoidal()
        self.loss      = BinaryCrossentropy()

        self.x = Tensor([[1], [3], [4], [5], [8], [7]])
        self.y = Tensor([[0], [0], [0], [1], [1], [1]])

        self.w = Tensor(self.bacnkend.array_zeros((self.x.data.shape[1]), self.y.data.shape[1]), requires_grad=True)
        self.b = Tensor([0], requires_grad=True)

        self.doorstep = 0.5

    def learning(self, epochs):
        for epoch in range(epochs):
            logit = self.x * self.w + self.b
            probability = self.sigmoid.forward(self.x)
            loss = self.loss.forward(probability, self.y)

            self.optimizer.zero_grad([self.w, self.b])
            self.engine.backward(loss)
            self.optimizer.step()

            if (epoch + 1) % 100:
                print(f'||  Loss  {loss}  ||')

    def predict(self, testDigitX) -> int:
        logit = self.w * testDigitX + self.b
        probability = self.sigmoid.backward(logit)
        predict = (probability >= self.doorstep)

        return 1 if predict else 0

model = Classification()
model.predict(4)
        

