from ...autograd._operations.loss import bce
from ...backend.backend_manager import BackendManager
from .loss import Loss


class BinaryCrossentropy(Loss):
    def __init__(self):
        super().__init__(self)
        self.backend = BackendManager.get_backend()
        self.loss_op = bce.BinaryCrossentropy(self.backend)