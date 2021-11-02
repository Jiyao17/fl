
from torch import nn

from utils.tasks import Task

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
        for i in range(self.task.configs.l_epoch_num):
            self.task.train()

        return 0

    def test_model(self) -> float:
        pass
        # return self.task.test()
