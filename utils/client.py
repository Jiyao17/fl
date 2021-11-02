
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
        
    def train_model(self, l_epoch_num: int=0) -> float:
        if l_epoch_num != 0:
            for i in range(l_epoch_num):
                self.task.train()
        else:
            for i in range(self.task.configs.l_epoch_num):
                self.task.train()

        return 0

    def test_model(self) -> float:
        pass
        # return self.task.test()
