
import copy

from torch import nn
from torchvision.datasets.mnist import FashionMNIST as FMNIST

from utils.tasks import Task, TaskFashionMNIST
from utils.funcs import Config

class Client():
    def __init__(self, 
        task_name: int, 
        dataset: int, 
        epoch_num: int, 
        batch_size: int, 
        lr: float, 
        device: str,
        ):
        self.task_name = task_name
        self.task = None
        if self.task_name == "FashionMNIST":
            self.task: Task = TaskFashionMNIST(
                task_name, dataset, epoch_num, batch_size, lr, device)
        self.device = device

    def get_model(self) -> nn.Module:
        return self.task.model

    def update_model(self, global_model: nn.Module):
        state_dict = global_model.state_dict()
        self.task.model.load_state_dict(state_dict)
        
    def train_model(self) -> float:
        return self.task.train()

    def test_model(self) -> float:
        pass
        # return self.task.test()
