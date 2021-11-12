from typing import List, Tuple, overload
import copy
import os
from numpy.lib.function_base import select

import torch
from torch import nn, optim, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchvision import datasets, transforms
from torch import randperm
# SpeechCommand
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import Resample
import torch.nn.functional as F
# AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS 
from torchtext.data.functional import to_map_style_dataset

from utils.models import FashionMNIST, SpeechCommand, AGNEWS




class Config:

    TEST_TYPES = ["iid", "iid-range", "noniid-sigma", "noniid-sigma-group", "noniid-r", "noniid-group"]

    def __init__(self,
        task_name: str,
        g_epoch_num: int,
        client_num: int,
        l_data_num: int,
        l_epoch_num: int,
        l_batch_size: int,
        l_lr: int,
        datapath: int,
        device: int,
        result_dir: str,
        verbosity: int,
        simulation_num: int,
        reside: int=0,
        simulation_index: int=0,
        l_trainset: Dataset=None,
        testset: Dataset=None,
        sigma: float=0.1,
        test_type: str="iid",
        ) -> None:

        self.task_name: str = task_name
        # global parameters
        self.g_epoch_num: int = g_epoch_num
        # local parameters
        self.client_num: int = client_num
        self.l_data_num: int = l_data_num
        self.l_epoch_num: int = l_epoch_num
        self.l_batch_size: int = l_batch_size
        self.l_lr: float = l_lr
        # shared settings
        self.datapath: str = datapath
        self.device: torch.device = torch.device(device)
        self.result_dir: str = result_dir
        self.verbosity:int = verbosity

        # run multiple simulations in processes at one time
        self.simulation_num: int = simulation_num
        # for single simulators to know its index
        # so it can write results to its file
        self.simulation_index:int = simulation_index

        # task reside on server (-1) or client (0, 1, ..., client_num-1)
        self.reside:int = reside
        # this should be different for every client
        self.l_trainset: Dataset = l_trainset
        # this should be used by the server
        self.testset: Dataset = testset
        # non-IID degree
        self.sigma: int = sigma
        self.test_type: str = test_type

    def __init__(self):
        if len(UniTask.supported_tasks) < 1:
            raise "No supported task, cannot run"
        self.task_name: str = UniTask.supported_tasks[0]
        self.g_epoch_num: int = 100
        self.client_num: int = 100
        self.l_data_num: int = 500
        self.l_epoch_num: int = 5
        self.l_batch_size: int = 64
        self.l_lr: float = 0.01
        self.datapath: str = "./data/"
        self.device: torch.device = torch.device("cuda")
        self.result_dir: str = "./result/"
        self.verbosity:int = 2
        self.simulation_num: int = 1
        self.reside:int = -1
        self.simulation_index:int = 0
        self.l_trainset: Dataset = None
        self.testset: Dataset = None
        self.sigma: float = 0.1
        self.test_type: str = "iid"


class Task:
    # test dataloader for tasks on server
    # train dataloader for tasks on clients
    # init once for every simulation
    testset: Dataset = None
    trainset: Dataset = None
    trainset_perm: 'list[int]' = None

    def __init__(self, configs: Config):
        self.configs = configs

        self.model: nn.Module = None
        self.train_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None
        self.optimizer: Optimizer = None
        self.scheduler = None

    @overload
    def get_dataloader(self):
        """
        Initialize static members
        """
        pass

    @overload
    def train(self) -> float:
        pass

    @overload
    def test(self) -> float:
        pass

    def get_model(self) -> nn.Module:
        return self.model.to(self.configs.device)

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
    def get_datasets(config: Config) -> Tuple[Dataset, Dataset]:
        testset = datasets.FashionMNIST(
            root=config.datapath,
            train=False,
            # download=True,
            transform=transforms.ToTensor(),
            )
        trainset = datasets.FashionMNIST(
            root=config.datapath,
            train=True,
            # download=True,
            transform=transforms.ToTensor(),
            )

        return (trainset, testset)

    def __init__(self, configs: Config):
        super().__init__(configs)

        self.model = FashionMNIST()
        self.loss_fn = nn.modules.loss.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.configs.l_lr)
        self.get_dataloader()

    def get_dataloader(self):

        # if dataset not loaded, load first
        # if Task.trainset_perm == None:
        #     Task.trainset_perm = randperm(len(Task.trainset)).tolist()

        self.testset = self.configs.testset
        self.test_dataloader = DataLoader(
            self.testset,
            batch_size=self.configs.l_batch_size,
            shuffle=False,
            drop_last=True
        )

        if 0 <= self.configs.reside and self.configs.reside <= self.configs.client_num:
        #     data_num = self.configs.l_data_num
        #     reside = self.configs.reside
        #     self.trainset = Subset(Task.trainset,
        #         Task.trainset_perm[data_num*reside: data_num*(reside+1)])
            self.trainset = self.configs.l_trainset
        # print(len(self.trainset))
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
                    (self.configs.simulation_index, len(self.configs.l_trainset)))

    def train(self) -> float:
        self.model.to(self.configs.device)
        self.model.train()

        for X, y in self.train_dataloader:
            # Compute prediction and loss
            pred = self.model(X.to(self.configs.device))
            # print(y.shape)
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
            test_loss += self.loss_fn(pred, y.to(self.configs.device)).item()
            correct += (pred.argmax(1) == y.to(self.configs.device)).type(torch.float).sum().item()
        
        correct /= 1.0*size
        test_loss /= 1.0*size

        return correct, test_loss


class TaskSpeechCommand(Task):

    labels: list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
        'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
        'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']

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

    def __init__(self, configs: Config):
        super().__init__(configs)

        self.model = SpeechCommand()
        self.loss_fn = F.nll_loss
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.configs.l_lr)
        self.get_dataloader()

        waveform, sample_rate, label, speaker_id, utterance_number = self.testset[0]
        new_sample_rate = 8000
        transform = Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        # transformed: Resample = transform(waveform)
        self.transform = transform.to(self.configs.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs.l_lr, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)  # reduce the learning after 20 epochs by a factor of 10

    @staticmethod
    def get_datasets(config: Config) -> Tuple[Dataset, Dataset]:
        testset = TaskSpeechCommand.SubsetSC("testing", config.datapath)
        trainset = TaskSpeechCommand.SubsetSC("training", config.datapath)

        return (trainset, testset)


    def get_dataloader(self):

        if self.configs.device == torch.device("cuda"):
            num_workers = 1
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False

        # test dataloader
        self.testset = self.configs.testset
        self.test_dataloader = DataLoader(
                self.testset,
                batch_size=self.configs.l_batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=TaskSpeechCommand.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                )
        # train dataloader
        if 0 <= self.configs.reside and self.configs.reside <= self.configs.client_num:
        #     data_num = self.configs.l_data_num
        #     reside = self.configs.reside
        #     self.trainset = Subset(Task.trainset,
        #         Task.trainset_perm[data_num*reside: data_num*(reside+1)])
            self.trainset = self.configs.l_trainset
            self.train_dataloader = DataLoader(
                self.trainset,
                batch_size=self.configs.l_batch_size,
                shuffle=True,
                collate_fn=TaskSpeechCommand.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True
                )

    def train(self):
        self.model.to(self.configs.device)
        self.model.train()
        self.transform = self.transform.to(self.configs.device)
        for data, target in self.train_dataloader:
            data = data.to(self.configs.device)
            target = target.to(self.configs.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = self.loss_fn(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

    def test(self):
        self.model.to(self.configs.device)
        self.model.eval()

        dataset_size = len(self.test_dataloader.dataset)
        correct, loss = 0, 0
        for data, target in self.test_dataloader:
            data = data.to(self.configs.device)
            target = target.to(self.configs.device)
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            pred = TaskSpeechCommand.get_likely_index(output)
            loss += self.loss_fn(output.squeeze(), target).item()

            # pred = output.argmax(dim=-1)
            correct += TaskSpeechCommand.number_of_correct(pred, target)

        correct /= 1.0*dataset_size
        loss /= 1.0*dataset_size

        return correct, loss

    @staticmethod
    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(TaskSpeechCommand.labels.index(word))

    @staticmethod
    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return TaskSpeechCommand.labels[index]

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
            targets += [TaskSpeechCommand.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = TaskSpeechCommand.pad_sequence(tensors)
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


class TaskAGNEWS(Task):
    def __init__(self, configs: Config):
        super().__init__(configs)
        self.get_dataloader()
        self.tokenizer = get_tokenizer('basic_english')
        self.train_iter = AG_NEWS(root=self.configs.datapath, split='train')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x) - 1

        self.model = AGNEWS()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configs.l_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)

        self.get_dataloader()

    @staticmethod
    def get_datasets(config: Config) -> Tuple[Dataset, Dataset]:
        test_iter = AG_NEWS(root=config.datapath, split="test")
        testset = to_map_style_dataset(test_iter)   
        train_iter = AG_NEWS(root=config.datapath, split="train")
        trainset = to_map_style_dataset(train_iter)

        return (trainset, testset)


    def get_dataloader(self):
        self.testset = self.configs.testset
        self.trainset = self.configs.l_trainset

        self.test_dataloader = DataLoader(
            self.testset,
            batch_size=self.configs.l_batch_size,
            shuffle=True, 
            collate_fn=self.collate_batch)

        self.train_dataloader = DataLoader(
            self.trainset, 
            batch_size=self.configs.l_batch_size, 
            shuffle=False, 
            collate_fn=self.collate_batch)

    def train(self) -> float:
        self.model.to(self.configs.device)
        self.model.train()
        total_acc, total_count = 0, 0

        for label, text, offsets in self.train_dataloader:
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.loss_fn(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        
        return total_acc/total_count

    def test(self) -> float:
        self.model.to(self.configs.device)
        self.model.eval()

        total_acc, loss = 0, 0
        with torch.no_grad():
            for label, text, offsets in self.test_dataloader:
                predicted_label = self.model(text, offsets)
                loss += self.loss_fn(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
        
        size = len(self.test_dataloader.dataset)
        total_acc /= 1.0*size
        loss /= 1.0*size

        return total_acc, loss

    def yield_tokens(self, data_iter):
        # return [self.tokenizer(text) for _, text in data_iter]
        for _, text in data_iter:
            yield self.tokenizer(text)

        # def transform_to_token(self, data)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        device = self.configs.device
        return label_list.to(device), text_list.to(device), offsets.to(device)


class UniTask:
    """
    Use UniTask().get_task() to get correct task type
    """
    #  "AG_NEWS"
    supported_tasks = ["FashionMNIST", "SpeechCommand", "AG_NEWS"]

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_task(config: Config) -> Task:
        if config.task_name not in UniTask.supported_tasks:
            raise "Task not supported yet."

        if config.task_name == "FashionMNIST":
            task = TaskFashionMNIST(config)
        if config.task_name == "SpeechCommand":
            task = TaskSpeechCommand(config)
        if config.task_name == "AG_NEWS":
            task = TaskAGNEWS(config)
        return task

    def get_datasets(config: Config) -> Tuple[Dataset, Dataset]:
        if config.task_name == "FashionMNIST":
            trainset, testset = TaskFashionMNIST.get_datasets(config)
        if config.task_name == "SpeechCommand":
            trainset, testset = TaskSpeechCommand.get_datasets(config)
        if config.task_name == "AG_NEWS":
            trainset, testset = TaskAGNEWS.get_datasets(config)
        
        return (trainset, testset)