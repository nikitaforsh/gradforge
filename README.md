<h1 align="center">
  <img src="https://github.com/insane-machines/fsh/blob/main/fsh/fsh.egg-info/logo.jpg"></img>
</h1>
<h1 align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=00FFAA&center=true&vCenter=true&width=500&lines=Hi,+I'm+nikitaforsh!;Welcome+to++our+machinelearning+library;named+GradForge!" alt="Typing SVG" />
</h1>

<p align="center">
  <a href="https://t.me/insane_machines" target="_blank">
    <img src="https://img.shields.io/badge/Insane%20Machines-Telegram-blue?style=for-the-badge&logo=telegram" alt="Telegram link" />
  </a>
</p>
Features

Models and layers with: 
• **Autograd**  
• **Callbacks**
• **Metrics** 
• **Preprocessing functions**

---
**Project published on PyPI**

Install it with pip

```bash
pip install gradforge
```

Quick Start

```python
from gradforge.core import Tensor
from gradforge.optimizers import SGD
from gradforge.autograd.standard_engine import StandardEngine
from gradforge.backend import BackendManager, NumpyBackend
from gradforge.training.layers.linear import Linear

BackendManager.set_backend(NumpyBackend())

x       = Tensor([[4], [8], [9]])
y       = Tensor([[8], [16], [18]])
test    = Tensor([[40], [60]])

model = Linear(1, 1)
engine = StandardEngine()
optimizer = SGD(learning_rate=1e-3)
for epoch in range(1000):
    y_pred  = model.predict(x)
    error   = y - y_pred
    loss    = error ** 2

    optimizer.zero_grad(model.params)
    engine.backward(loss)
    optimizer.step(model.params)

print(model.predict(test).data)
