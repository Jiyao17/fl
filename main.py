
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text

from io import TextIOWrapper
from multiprocessing.context import Process

import torch
from utils.simulator import Simulator
from utils.server import Server
from utils.client import Client

import lzma

if __name__ == "__main__":

    simulator: Simulator = Simulator()
    simulator.parse_args()

    a = lzma.CHECK_CRC32

    print(simulator.configs.l_data_num)


