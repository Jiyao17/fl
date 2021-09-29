
from utils.tasks import Task


class Client():
    def __init__(self, task: Task):
        self.task = task
        
    def train_model(self) -> float:
        return self.task.train()

    def test_model(self) -> float:
        return self.task.test()
