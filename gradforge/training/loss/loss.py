from ...autograd.operation import Operation
from ...core.tensor import Tensor


class Loss:
    def __init__(self) -> None:
        super().__init__()
        self.loss_op: Operation | None = None
        
    def predict(self, data: list[Tensor]):
        result_tensor = Tensor._form_result(self.loss_op, 
                                            [e.data for e in data], 
                                            data, 
                                            requires_grad=any(element.requires_grad for element in data))
        return result_tensor 