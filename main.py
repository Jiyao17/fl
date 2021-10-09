
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

import json
from multiprocessing import Process, get_start_method, set_start_method

from utils.models import FashionMNIST
from utils.simulator import Simulator
from utils.tasks import Config, UniTask


if __name__ == "__main__":
    # if get_start_method(False) != "spawn":
    set_start_method("spawn")
    client_nums = [5]

    simulator: Simulator = Simulator()
    simulator.configs.task_name = UniTask.supported_tasks[1]
    simulator.configs.l_data_num = 6000
    simulator.configs.g_epoch_num = 1000
    simulator.configs.simulation_num = 5
    for i in range(0, 3):
        simulator.configs.task_name = UniTask.supported_tasks[i]
        for j in client_nums:
            simulator.configs.client_num = j
            simulator.start()

    # print(simulator.configs.l_data_num)


