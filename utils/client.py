
import copy

from torch import nn
from torch.utils.data import Dataset

from utils.tasks import Task, TaskFashionMNIST
from utils.configs import Config

class Client():
    def __init__(self, task: Task):
        self.task = task

    def get_model(self) -> nn.Module:
        return self.task.get_model()

    def update_model(self, global_model: nn.Module):
        # state_dict = global_model.state_dict()
        self.task.update_model(global_model)
        # self.task.model.load_state_dict(state_dict)
        
    def train_model(self) -> float:
        return self.task.train()

    def test_model(self) -> float:
        pass
        # return self.task.test()
