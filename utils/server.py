
import copy

import torch
from torch import nn

from utils.client import Client
from utils.tasks import Task

class Server():
    @staticmethod
    def group_distribute(model: nn.Module, clients: 'list[Client]'):
        for client in clients:
            client.update_model(model)
    
    @staticmethod
    def group_aggregate(clients: 'list[Client]'):
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
        
        return state_dict_avg

    def global_aggregate(self, models: 'list[nn.Module]'):
        """
        only for grouped train
        """
        state_dicts = [
            model.state_dict()
            for model in models
            ]
        state_dict_avg = copy.deepcopy(state_dicts[0]) 
        for key in state_dict_avg.keys():
            for i in range(1, len(state_dicts)):
                state_dict_avg[key] += state_dicts[i][key]
            state_dict_avg[key] = torch.div(state_dict_avg[key], len(state_dicts))
        
        self.task.load_state_dict(state_dict_avg)

    def __init__(self, task: Task):
        # self.task = copy.deepcopy(task)
        self.task = task
        self.configs = self.task.configs

    def distribute_model(self, clients: 'list[Client]'):
        for client in clients:
            client.update_model(self.task.get_model())

    def test_model(self) -> float:
        return self.task.test()

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
        
        self.task.load_state_dict(state_dict_avg)

    def test_model(self) -> float:
        return self.task.test()

    # def reset_model(self):
    #     dic = copy.deepcopy(self.init_model_dict)
    #     self.model.load_state_dict(dic)
    #     del(dic)




