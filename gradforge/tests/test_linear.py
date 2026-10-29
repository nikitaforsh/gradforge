#import tracemalloc

import cProfile
import pstats

from gradforge.autograd import StandardEngine

#from gradforge.backend import BackendManager
from gradforge.core import Tensor
from gradforge.optimizers import SGD
from gradforge.training.layers import Linear

#tracemalloc.start()

def test_train():
    #before = tracemalloc.take_snapshot()

    x       = Tensor([[4], [8], [9]])
    y       = Tensor([[8], [16], [18]])
    #test    = Tensor([[40], [60]])
    model = Linear(1, 1) 
    assert x.data.shape == y.data.shape 
    engine = StandardEngine()
    optimizer = SGD(learning_rate=1e-3)
    for epoch in range(10000):
        y_pred  = model.predict(x)
        error   = y - y_pred
        loss    = error ** 2

        optimizer.zero_grad(model.params)
        engine.backward(loss)
        optimizer.step(model.params)

    #print(test.data @ model.w.data)

    #after = tracemalloc.take_snapshot()
 

    #stats = after.compare_to(before, "lineno")
    #for result in stats[:50]:
    #    print(result)

profiler = cProfile.Profile()
profiler.enable()
test_train()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("tottime")
stats.print_stats(30)
# 0.757 seconds
# Backend allocations needs to optimize