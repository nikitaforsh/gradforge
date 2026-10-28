from ...autograd.operation import Operation
from ...core.tensor import Tensor


class Loss:
    def __init__(self) -> None:
        super().__init__(self)
        self.loss_op: Operation | None = None
        
    def predict(self, data: Tensor):
        result_tensor = data._form_result(self.loss_op, [data.data], [data], requires_grad=data.requires_grad)
        return result_tensor