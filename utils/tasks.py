
from typing import overload
import copy

from torch import nn, optim
from torch.optim.optimizer import Optimizer

from torch.utils.data import Dataset, DataLoader


from utils.models import FashionMNIST, SpeechCommand, AG_NEWS


class Task:
    def __init__(self,
        task_name: int,
        dataset: Dataset,
        epoch_num: int,
        batch_size: int,
        lr: float,
        device: str,
    ):
        self.task_name = task_name
        self.dataset = dataset
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.model: nn.Module = None
        self.dataloader: DataLoader = None
        self.optimizer: Optimizer = None
        self.scheduler = None

    def train(self):
        pass

    def get_model(self) -> nn.Module:
        return self.model

    def update_model(self, new_model: nn.Module):
        state_dict = new_model.state_dict()
        new_state_dict = copy.deepcopy(state_dict)
        self.model.load_state_dict(new_state_dict)

class TaskFashionMNIST(Task):
    def __init__(self, 
        task_name: int, 
        dataset: int, 
        epoch_num: int, 
        batch_size: int, 
        lr: float, 
        device: str,
        optimizer=None,
        scheduler=None,
        ):

        super().__init__(task_name, dataset, epoch_num, batch_size,
            lr, device, optimizer, scheduler)

        self.model = FashionMNIST()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
            )
        self.loss_fn = nn.modules.loss.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)


    @overload
    def train(self):
        self.model.train()

        for (X, y) in self.dataloader:
            X.to(self.device)
            y.to(self.device)
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

