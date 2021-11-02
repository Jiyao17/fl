
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

import json
from multiprocessing import set_start_method

from utils.simulator import Simulator
from utils.tasks import UniTask


if __name__ == "__main__":
    # if get_start_method(False) != "spawn":
    set_start_method("spawn")
    # client_nums = [3, 4, 5, 6, 7]
    # lrs = [0.01, 0.01, 0.1]

    simulator: Simulator = Simulator()
    simulator.configs.task_name = UniTask.supported_tasks[0]
    simulator.configs.client_num = 100
    simulator.configs.l_data_num = 500
    simulator.configs.g_epoch_num = 100
    simulator.configs.sigma = 0.8
    simulator.configs.simulation_num = 5

    # for i, task_name in enumerate(UniTask.supported_tasks):
        # simulator.configs.task_name = task_name
        # simulator.configs.l_lr = 0.01

    # for client_num in client_nums:
        # simulator.configs.client_num = client_num
    
    simulator.start()



    # print(simulator.configs.l_data_num)


