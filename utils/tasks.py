
from typing import overload
import copy
import os

import torch
from torch import nn, optim, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset, Subset
from torchvision import datasets, transforms
from torch import randperm
from torchaudio.datasets import SPEECHCOMMANDS

from utils.configs import Config
from utils.models import FashionMNIST, SpeechCommand, AG_NEWS


class Task:
    # test dataloader for tasks on server
    # train dataloader for tasks on clients
    # init once for every simulation
    testset: Dataset = None
    trainset: Dataset = None
    trainset_perm: 'list[int]' = None

    @staticmethod
    @overload
    def get_dataset(configs: Config):
        """
        Initialize static members
        """
        pass


    def __init__(self, configs: Config):
        self.configs = configs

        self.model: nn.Module = None
        self.dataloader: DataLoader = None
        self.optimizer: Optimizer = None
        self.scheduler = None

    def get_dataloader(self):
        self.testset = TaskFashionMNIST.testset
        self.test_dataloader = DataLoader(
            self.testset,
            batch_size=self.configs.l_batch_size,
            shuffle=False,
            drop_last=True
        )

        if 0 <= self.configs.reside and self.configs.reside <= self.configs.client_num:
            data_num = self.configs.l_data_num
            reside = self.configs.reside
            self.trainset = Subset(TaskFashionMNIST.trainset,
                TaskFashionMNIST.trainset_perm[data_num*reside: data_num*(reside+1)])
        self.train_dataloader = DataLoader(
                self.trainset,
                batch_size=self.configs.l_batch_size,
                shuffle=True,
                drop_last=True
                )

        if self.configs.verbosity >= 3:
            if self.configs.reside == -1:
                print("Test set length in simulation %d: %d" %
                    (self.configs.simulation_index, len(self.testset)))
            else:
                print("Dataset length in simulation %d: %d, %d-%d" %
                    (self.configs.simulation_index, data_num, data_num*reside, data_num*(reside+1)))

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
    def get_dataset(configs: Config):
        # if dataset not loaded, load first
        if Task.testset == None:
            Task.testset = datasets.FashionMNIST(
                root=configs.datapath,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
                )
        if Task.trainset == None:
            Task.trainset = datasets.FashionMNIST(
                root=configs.datapath,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
                )
        if Task.trainset_perm == None:
            Task.trainset_perm = randperm(len(TaskFashionMNIST.trainset)).tolist()

    def __init__(self, configs: Config):
        super().__init__(configs)

        self.model = FashionMNIST()
        self.loss_fn = nn.modules.loss.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.configs.l_lr)
        TaskFashionMNIST.get_dataset(self.configs)
        self.get_dataloader()

    def train(self) -> float:
        self.model.to(self.configs.device)
        self.model.train()

        for X, y in self.train_dataloader:
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

        size = len(self.testset)
        test_loss, correct = 0, 0
        # with torch.no_grad():
        for X, y in self.test_dataloader:
            pred = self.model(X.to(self.configs.device))
            # test_loss += loss_fn(pred, y.to(self.device)).item()
            correct += (pred.argmax(1) == y.to(self.configs.device)).type(torch.float).sum().item()
        correct /= size

        return correct


class TaskSpeechCommand(Task):

    labels = None

    @staticmethod
    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(SpeechCommand.labels.index(word))

    @staticmethod
    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return SpeechCommand.labels[index]

    @staticmethod    
    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    @staticmethod
    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [SpeechCommand.label_to_index(label)]

        # Group the list of tensors into a batched tensor
            tensors = SpeechCommand.pad_sequence(tensors)
            targets = torch.stack(targets)

            return tensors, targets

    @staticmethod
    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    @staticmethod
    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    class SubsetSC(SPEECHCOMMANDS):
        def __init__(self, subset, data_path):
            super().__init__(root=data_path, download=True)

            def load_list(filename):
                filepath = os.path.join(self._path, filename)
                with open(filepath) as fileobj:
                    return [os.path.join(self._path, line.strip()) for line in fileobj]

            if subset == "validation":
                self._walker = load_list("validation_list.txt")
            elif subset == "testing":
                self._walker = load_list("testing_list.txt")
            elif subset == "training":
                excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                excludes = set(excludes)
                self._walker = [w for w in self._walker if w not in excludes]


    @staticmethod
    def get_dataset(configs: Config):
        # if dataset not loaded, load first
        if Task.testset == None:
            Task.testset = datasets.FashionMNIST(
                root=configs.datapath,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
                )
        if Task.trainset == None:
            Task.trainset = datasets.FashionMNIST(
                root=configs.datapath,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
                )
        if Task.trainset_perm == None:
            Task.trainset_perm = randperm(len(TaskFashionMNIST.trainset)).tolist()

    def __init__(self, configs: Config):
        super().__init__(configs)

        self.model = SpeechCommand()
        self.loss_fn = nn.modules.loss.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.configs.l_lr)
        TaskSpeechCommand.get_dataset(self.configs)
        self.get_dataloader()




class UniTask:
    """
    Use UniTask().get_task() to get correct task type
    """
    #  "AG_NEWS"
    supported_tasks = ["FashionMNIST", "SpeechCommand",]

    def __init__(self, configs: Config) -> None:
        self.configs = copy.deepcopy(configs)
        self.task = None
        if self.configs.task_name not in UniTask.supported_tasks:
            raise "Task not supported yet."

        if self.configs.task_name == "FashionMNIST":
            self.task = TaskFashionMNIST(self.configs)
        if self.configs.task_name == "SpeechCommand":
            self.task = SpeechCommand(self.configs)

    def get_task(self) -> Task:
        return self.task