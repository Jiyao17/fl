
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

from multiprocessing import Process, get_start_method, set_start_method

from utils.models import FashionMNIST
from utils.simulator import Simulator
from utils.tasks import Config


if __name__ == "__main__":
    # if get_start_method(False) != "spawn":
    set_start_method("spawn")

    simulator: Simulator = Simulator()
    simulator.configs.l_data_num = 5000
    simulator.configs.g_epoch_num = 500
    simulator.configs.simulation_num = 5

    # for i in range(1, 12):
    simulator.configs.client_num = 12
        # simulator.check_configs()
    simulator.start()


    # print(simulator.configs.l_data_num)


