
from typing import overload
import copy

import torch
from torch import nn, optim, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.configs import Config
from utils.models import FashionMNIST, SpeechCommand, AG_NEWS


class Task:
    # test dataloader for tasks on server
    # train dataloader for tasks on clients
    dataloader = None

    @overload
    @staticmethod
    def get_dataloader(configs: Config, reside: int):
        pass

    def __init__(self, configs: Config, index: int):
        self.configs = configs
        # -1: task is on server
        # non-neg int: client No.
        self.index = index

        self.model: nn.Module = None
        self.dataloader: DataLoader = None
        self.optimizer: Optimizer = None
        self.scheduler = None

    @overload
    def train(self) -> float:
        pass

    @overload
    def test(self) -> float:
        pass

    def get_model(self) -> nn.Module:
        return self.model

    def update_model(self, new_model: nn.Module):
        state_dict = new_model.state_dict()
        new_state_dict = copy.deepcopy(state_dict)
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.configs.device)

    def load_state_dict(self, new_state_dict: 'dict[str, Tensor]'):
        state_dict = copy.deepcopy(new_state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.configs.device)

class TaskFashionMNIST(Task):

    @staticmethod
    def get_dataloader(configs: Config, reside: int):
        if TaskFashionMNIST.dataloader == None:
            if reside == -1:
                dataset = datasets.FashionMNIST(
                    root=configs.datapath,
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                    )
            elif reside >= 0:
                dataset = datasets.FashionMNIST(
                    root=configs.datapath,
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                    )
                data_num = configs.l_data_num
                dataset = dataset[data_num*reside:data_num*(reside+1)]
            else:
                raise "Unknow reside"

            TaskFashionMNIST.dataloader = DataLoader(
                    dataset,
                    batch_size=configs.l_batch_size,
                    shuffle=True,
                    drop_last=True
                    )

    def __init__(self, configs: Config, index: int):
        super().__init__(configs, index)

        self.model = FashionMNIST()
        self.loss_fn = nn.modules.loss.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.configs.l_lr)
        if TaskFashionMNIST.dataloader == None:
            TaskFashionMNIST.get_dataloader(configs, index)

    def train(self) -> float:
        self.model.to(self.configs.device)
        self.model.train()

        for X, y in TaskFashionMNIST.dataloader:
            # Compute prediction and loss
            pred = self.model(X.to(self.configs.device))
            loss = self.loss_fn(pred, y.to(self.configs.device))
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return 0

    def test(self) -> float:
        self.model.to(self.configs.device)
        self.model.eval()

        size = len(TaskFashionMNIST.dataloader.dataset)
        test_loss, correct = 0, 0
        # with torch.no_grad():
        for X, y in TaskFashionMNIST.dataloader:
            pred = self.model(X.to(self.configs.device))
            # test_loss += loss_fn(pred, y.to(self.device)).item()
            correct += (pred.argmax(1) == y.to(self.configs.device)).type(torch.float).sum().item()
        correct /= size

        return correct


class UniTask:
    """
    Use UniTask().get_task() to get correct task type
    """
    supported_tasks = ["FashionMNIST", "SpeechCommand", "AG_NEWS"]

    def __init__(self, configs: Config, index: int) -> None:
        self.configs = configs
        self.index = index
        self.task = None
        if self.configs.task_name not in UniTask.supported_tasks:
            raise "Task not supported yet."

        if self.configs.task_name == "FashionMNIST":
            self.task = TaskFashionMNIST(self.configs, self.index)

    def get_task(self) -> Task:
        return self.task