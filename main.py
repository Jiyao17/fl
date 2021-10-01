
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

def run_sim(que: Queue, progress_file: str, task_name, g_epoch_num, client_num, l_data_num, l_epoch_num, l_batch_size, l_lr, data_path, device, verbosity):
    # partition data
    datasets = get_partitioned_datasets(task_name, client_num, l_data_num, l_batch_size, data_path)
    test_dataset = get_test_dataset(task_name, data_path)
    # initialize server and clients
    clients: list[Client] = [
        Client(task_name, datasets[i], l_epoch_num, l_batch_size, l_lr, device) 
        for i in range(client_num)
        ]
    
    server = Server(task_name, test_dataset, clients, g_epoch_num, device)

    result: list[float] = []
    g_accuracy = server.test_model()
    if verbosity >= 1:
        print(f"Global accuracy:{g_accuracy*100:.9f}%")
    for i in range(server.epoch_num):
        
        if verbosity >= 1:
            print("Epoch %d ......" % i)

        server.distribute_model()
        for j in range(len(server.clients)):
            # acc = server.clients[j].test_model()
            # print("before training: client %d accuracy %.9f at epoch %d" % (j, acc, i))
            server.clients[j].train_model()
            # acc = server.clients[j].test_model()
            # print("after training:  client %d accuracy %.9f at epoch %d" % (j, acc, i))
        server.aggregate_model()
        # l_accuracy = [client.test_model() for client in server.clients]
        g_accuracy = server.test_model()

        if verbosity >= 1:
            pf = open(progress_file, "a")
            print(f"Global accuracy:{g_accuracy*100:.9f}%")
            pf.write(f"Epoch {i}: {g_accuracy*100:.2f}%\n")
            # if i % 10 == 9:
            pf.flush()
            pf.close()
            # print(f"Local accuracy after training: {[acc for acc in l_accuracy]}")
        
        if i % 10 == 9:
            result.append(g_accuracy)

    que.put(result)


if __name__ == "__main__":

    simulator: Simulator = Simulator()


