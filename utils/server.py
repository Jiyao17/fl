
import copy

import torch
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torchaudio

class Server():
    def __init__(self):

        self.init_task()

    def init_task(self):
        pass

    def distribute_model(self):
        """
        Send global model to clients.
        """

    def aggregate_model(self):
        state_dicts = [
            client.model.state_dict()
            for client in self.clients
            ]
        # calculate average model
        state_dict_avg = copy.deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.model.load_state_dict(state_dict_avg)

    def test_model(self) -> float:
        self.model = self.model.to(self.device)
        self.model.eval()

    def reset_model(self):
        dic = copy.deepcopy(self.init_model_dict)
        self.model.load_state_dict(dic)
        del(dic)




