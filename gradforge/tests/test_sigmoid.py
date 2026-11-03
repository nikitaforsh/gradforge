from gradforge.autograd.standard_engine import StandardEngine
from gradforge.core.tensor import Tensor
from gradforge.optimizers import SGD
from gradforge.training.layers import Linear
from gradforge.training.layers.sigmoid import Sigmoid
from gradforge.training.loss import BinaryCrossentropy


def test_binary():
    x = Tensor([[-4], [5]])
    test = Tensor([[6]])

    model = Linear(1, 1)
    probs = Sigmoid()
    loss = BinaryCrossentropy()
    engine = StandardEngine()
    optimizer = SGD(learning_rate=0.01)

    for e in range(100):
        linear_pred = model.predict(x)
        pred = probs.predict(linear_pred)
        loss_pred = loss.predict([pred, Tensor([[0], [1]])])

        optimizer.zero_grad(model.params)
        engine.backward(loss_pred)
        optimizer.step(model.params)

    linear_pred = model.predict(test)
    pred = probs.predict(linear_pred)

    assert any(pred.data) == True

