from gradforge.autograd.standard_engine import StandardEngine
from gradforge.core.tensor import Tensor
from gradforge.training.layers.sigmoid import Sigmoid

x = Tensor([-4, 5])
model = Sigmoid()
engine = StandardEngine()
pred = model.predict(x)
print(pred.data)