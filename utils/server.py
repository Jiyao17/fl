
import copy

import torch
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torchaudio
from client import Client

from utils.funcs import Config

class Server():
    def __init__(self, configs: Config):
        self.configs = configs

        self.model: nn.Module = None

    def distribute_model(self, clients: 'list[Client]'):
        state_dict = self.model.state_dict()
        for client in clients:
            client.get_model().load_state_dict(state_dict)
            client.get_model().to(client.device)

    def aggregate_model(self, clients: 'list[Client]'):
        state_dicts = [
            client.get_model().state_dict()
            for client in clients
            ]
        # calculate average model
        state_dict_avg = copy.deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)
        self.model.to(self.configs.device)

    def test_model(self) -> float:
        self.model = self.model.to(self.configs.device)
        self.model.eval()

    def reset_model(self):
        dic = copy.deepcopy(self.init_model_dict)
        self.model.load_state_dict(dic)
        del(dic)




