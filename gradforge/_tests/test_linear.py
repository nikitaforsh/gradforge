from gradforge.core.tensor import Tensor
from gradforge.optimizers.sgd import SGD
from gradforge.autograd.standard_engine import StandardEngine
from gradforge.backend.backend_manager import BackendManager
from gradforge.training.layers.linear import Linear
import tracemalloc

#tracemalloc.start()
    
def train(model, x, y, epochs):
    engine = StandardEngine()
    optimizer = SGD(learning_rate=1e-3)
    for epoch in range(epochs):
        y_pred  = model.predict(x)
        error   = y - y_pred
        loss    = error ** 2

        optimizer.zero_grad(model.params)
        engine.backward(loss)
        print('eror grad: ', error.grad)
        try:
            optimizer.step(model.params)
        except Exception as e:
            print('Error: ', e)

    print(x.data @ model.w.data)

x       = Tensor([[4], [8], [9]])
y       = Tensor([[8], [16], [18]])
test    = Tensor([[40], [60]])
assert x.data.shape == y.data.shape

#before = tracemalloc.take_snapshot()

model = Linear(1, 1)
train(model, x, y, epochs=10000)
# after = tracemalloc.take_snapshot()

print(test.data @ model.w.data)
# print(model.w.data.shape)

# stats = after.compare_to(before, "lineno")
# for result in stats[:50]:
#     print(result)

# 0.757 seconds
# Backend allocations needs to optimize