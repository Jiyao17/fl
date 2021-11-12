
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

import json
from multiprocessing import set_start_method
from warnings import simplefilter

from utils.simulator import Simulator
from utils.tasks import Task, UniTask

def iid_test():
    sigmas = [0.9, 0.8, 0.6, 0.5, 0.3, 0.2]

    for sigma in sigmas:
        simulator: Simulator = Simulator()
        simulator.configs.task_name = UniTask.supported_tasks[0]
        simulator.configs.client_num = 100
        simulator.configs.l_data_num = 600
        simulator.configs.l_epoch_num = 5
        simulator.configs.l_batch_size = 10
        simulator.configs.g_epoch_num = 500
        simulator.configs.sigma = sigma
        simulator.configs.simulation_num = 3
        
        simulator.configs.result_dir = "./result-noniid/"
        simulator.configs.test_type = "iid"

        simulator.start()

def opt_test():
    client_nums = [ 3, 4, 5, 6, 7 ]
    lrs = [0.005, 0.01, 0.1]
    result_dirs = ["./result-opt/", "./result-opt-1/", "./result-opt-2/"]

    # for i, task_name in enumerate(UniTask.supported_tasks):
    task_num = 2

    for client_num in client_nums:
        simulator: Simulator = Simulator()
        # simulator.configs.client_num = 100
        simulator.configs.l_data_num = 6000
        simulator.configs.l_epoch_num = 5
        simulator.configs.l_batch_size = 10
        simulator.configs.g_epoch_num = 500
        simulator.configs.sigma = -1
        simulator.configs.simulation_num = 3
        simulator.configs.result_dir = result_dirs[task_num]
        simulator.configs.test_type = "iid-range"
        
        simulator.configs.task_name = UniTask.supported_tasks[task_num]
        simulator.configs.l_lr = lrs[task_num]
        simulator.configs.client_num = client_num

        simulator.start()




if __name__ == "__main__":
    # prepare for multi-proccessing
    # if get_start_method(False) != "spawn":
    set_start_method("spawn")

    iid = 0
    if iid == 1:
        iid_test()
    else:
        opt_test()

    


    # print(simulator.configs.l_data_num)


