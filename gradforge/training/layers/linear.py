from .layer import Layer
from ...core.tensor import Tensor
from ...backend.backend_manager import BackendManager
class Linear(Layer):
    def __init__(self, in_shape, out_shape) -> None:
        super().__init__()      
        self._backend = BackendManager.get_backend()
        self.w: Tensor = Tensor(self._backend.array_zeros((in_shape, out_shape)), requires_grad=True)
        
    def predict(self, x: Tensor) -> Tensor:
        return x @ self.w

    @property
    def params(self) -> list[Tensor]:
        return [attr_val\
                    for attr_val in self.__dict__.values()\
                        if (type(attr_val) is Tensor and attr_val.requires_grad)]