
# code source:
# FashionMNIST: pytorch tutorial quickstart
# SpeechCommand: pytorch tutorial audio
# TextClassification: pytorch tutorial text


from utils.simulator import Simulator


if __name__ == "__main__":

    simulator: Simulator = Simulator()
    simulator.get_configs()

    simulator.start()


    # print(simulator.configs.l_data_num)


