from __future__ import annotations

from ..autograd._operations import add, div, matmul, mul, pow, sub
from ..autograd.operation import Operation
from ..backend.backend import Backend
from ..backend.backend_manager import BackendManager


class Tensor:
    __slots__ = ['backend', 'data', 'grad', 'grad_fn', 'grad_sum_axes', 'requires_grad']
    
    def __init__(self, data, requires_grad: bool = False, grad_sum_axes: tuple = (0, 0)):
        self.requires_grad: bool = requires_grad
        self.grad = 0
        self.grad_fn: list | None = None
        self.backend: Backend = BackendManager.get_backend()
        self.data = self.backend.convert(data)
        self.grad_sum_axes = grad_sum_axes

    def sum_grads(self):
        self.grad = self.backend.sum_array(self.grad, self.grad_sum_axes)

    def __mul__(self, other): 
        mul_op = mul.Mul(self.backend)
        
        return self._form_result(mul_op, [self.data, other.data], [self, other], requires_grad=\
                                            self.requires_grad or\
                                            other.requires_grad)

    def __matmul__(self, other: Tensor):
        mat_mul_op = matmul.Matmul(self.backend)
        return self._form_result(mat_mul_op, [self.data, other.data], [self, other], requires_grad=\
                                                        self.requires_grad or\
                                                        other.requires_grad)

    def __add__(self, other: Tensor):
        add_op = add.Add(self.backend)
        return self._form_result(add_op, [self.data, other.data], [self, other], requires_grad=\
                                                        self.requires_grad or\
                                                        other.requires_grad)

    def __sub__(self, other: Tensor):
        sub_op = sub.Sub(self.backend)
        return self._form_result(sub_op, [self.data, other.data], [self, other], requires_grad=\
                                                        self.requires_grad or\
                                                        other.requires_grad)

    def __truediv__(self, other):
        div_op = div.Div(self.backend)
        return self._form_result(div_op, [self.data, other.data], [self, other], requires_grad=\
                                                        self.requires_grad or\
                                                        other.requires_grad)

    def __pow__(self, other):
        pow_op = pow.Pow(self.backend)
        return self._form_result(pow_op, [self.data, other], [self], requires_grad=\
                                                        self.requires_grad)

    def _form_result(self, operation: Operation, data: list, parents: list, requires_grad=True):
        operation_forward = operation.forward(*data)

        result = Tensor(operation_forward, requires_grad=requires_grad)
        result.grad_fn = [parents, operation]

        return result

        