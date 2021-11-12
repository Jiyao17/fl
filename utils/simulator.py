
from argparse import ArgumentParser
import copy
from io import TextIOWrapper
from multiprocessing import Process
from os import confstr

import torch
from torch.utils.data.dataset import Dataset

from utils.client import Client
from utils.server import Server
from utils.tasks import UniTask, Config
from utils.data import dataset_split

def single_simulation(configs: Config):
    ssimulator = __SingleSimulator(configs)
    ssimulator.start()

class __SingleSimulator:
    def __init__(self, config: Config) -> None:
        self.config = config

        # set server
        trainset, testset = UniTask.get_datasets(self.config)
        self.config.reside = -1
        self.config.testset = testset
        server_task = UniTask.get_task(self.config)
        self.server = Server(server_task)

        self.clients: list[Client] = []
        datasets = dataset_split(trainset, self.config)
        for i in range(self.config.client_num):
            new_configs = copy.deepcopy(self.config)
            new_configs.reside = i
            new_configs.l_trainset = datasets[i]
            # print(len(new_configs.l_trainset))d
            # print("reside @ client %d in simu %d" % (new_configs.reside, new_configs.simulation_index))
            self.clients.append(Client(UniTask.get_task(new_configs)))

    # def split_datasets(self, trainset:Dataset):
    #             # set clients
    #     # split datasets
    #     if self.config.test_type == "noniid-sigma":
    #         # non-iid split
    #         print("Spliting data non-iid")
    #         datasets = dataset_split(trainset, self.config, 0)
    #     else:
    #         # uniformly split
    #         print("Spliting data uniformly")
    #         datasets = dataset_split(trainset, self.config, 1)


    def start(self):

        result_file = self.config.result_dir + \
            "/result" + str(self.config.simulation_index)
        result_file_loss = self.config.result_dir + \
            "/result" + str(self.config.simulation_index) + "loss"
        f = open(result_file, "a")
        f_loss = open(result_file_loss, "a")
        if self.config.verbosity >= 3:
            print("writing to file: %s, simu num: %d" % (result_file, self.config.simulation_index))
        args = "{:12} {:12} {:10} {:10} {:11} {:12} {:4} {:5}".format(
            self.config.task_name, self.config.g_epoch_num, 
            self.config.client_num, self.config.l_data_num, 
            self.config.l_epoch_num, self.config.l_batch_size, 
            self.config.l_lr, self.config.sigma)
        
        conf_str = "TASK          G_EPOCH_NUM CLIENT_NUM L_DATA_NUM " + \
            "L_EPOCH_NUM L_BATCH_SIZE L_LR SIGMA\n" + args + "\n"
        f.write(conf_str)
        f.flush()
        f_loss.write(conf_str)
        f_loss.flush()

        if self.config.test_type[:3] == "iid":
            self.regular_train(f, f_loss)  
            # self.grouped_train(f)
        else:
            self.grouped_train(f) 

        # finished
        f.write("\n")
        f.close()
        f_loss.write("\n")
        f_loss.close()

    def regular_train(self, f: TextIOWrapper, f_loss: TextIOWrapper, ):
        print("doing regular train")
        for i in range(self.config.g_epoch_num):
            self.server.distribute_model(self.clients)
            for client in self.clients:
                client.train_model()
            self.server.aggregate_model(self.clients)

            # record result
            if self.config.verbosity >=3:
                g_accu, g_loss = self.server.test_model()
                print("accuracy %f in simulation %d at global epoch %d" %
                    (g_accu, self.config.simulation_index, i))
            elif self.config.verbosity >=2:
                if i % 10 == 9:
                    g_accu, g_loss = self.server.test_model()
                    print("accuracy %f, loss %f in simulation %d at global epoch %d" %
                        (g_accu, g_loss, self.config.simulation_index, i))

            if i % 10 == 9:
                if self.config.verbosity <=1:
                    g_accu, g_loss = self.server.test_model()
                f.write("{:.5f} ".format(g_accu))
                f.flush()
                f_loss.write("{:.5f} ".format(g_loss))
                f_loss.flush()

    def grouped_train(self, f: TextIOWrapper, ):
        print("doing grouped train")

        targets = self.server.configs.testset.targets.tolist()
        category_num = len(set(targets))

        group_num = int(self.config.client_num/category_num)
        group_models = [self.server.task.model.to(self.config.device) for i in range(group_num) ]
        client_groups: 'list[list[Client]]' = [[] for i in range(group_num)]
        for i in range(group_num):
            client_groups[i] = self.clients[i*category_num : (i+1)*category_num]

        # used to check data distribution in any group
        # lable_on_clients = [[] for client in client_groups[6]]
        # for i, client in enumerate(client_groups[6]):
        #     for (sample, lable) in client.task.configs.l_trainset:
        #         lable_on_clients[i].append(lable)

        #     for j in range(10):
        #         a = lable_on_clients[i].count(j)
        #         print(a, end=" ")
        #     print("")
        
        for i in range(self.config.g_epoch_num):
            self.server.global_distribute(group_models)

            for j in range(group_num):
                Server.group_distribute(group_models[j], client_groups[j])
                for k in range(self.config.l_epoch_num):
                    for client in client_groups[j]:
                        client.train_model(1)
                group_state_dict = Server.group_aggregate(self.clients)
                group_models[j].load_state_dict(group_state_dict)
            
            self.server.global_aggregate(group_models)

            # record result
            if self.config.verbosity >=3:
                g_accu, g_loss = self.server.test_model()
                print("accuracy %f in simulation %d at global epoch %d" %
                    (g_accu, self.config.simulation_index, i))
            elif self.config.verbosity >=2:
                if i % 10 == 9:
                    g_accu, g_loss = self.server.test_model()
                    print("accuracy %f, loss %f in simulation %d at global epoch %d" %
                        (g_accu, g_loss, self.config.simulation_index, i))

            if i % 10 == 9:
                if self.config.verbosity <=1:
                    g_accu, g_loss = self.server.test_model()
                f.write("{:.5f} ".format(g_accu))
                f.flush()


class Simulator:

    @staticmethod
    def __get_argument_parser() -> ArgumentParser:
        ap = ArgumentParser()
        # positional
        ap.add_argument("task_name", type=str)
        ap.add_argument("g_epoch_num", type=int)
        ap.add_argument("client_num", type=int)
        ap.add_argument("l_data_num", type=int)
        ap.add_argument("l_epoch_num", type=int)
        ap.add_argument("l_batch_size", type=int)
        ap.add_argument("l_lr", type=float)
        # optional
        ap.add_argument("-p", "--datapath", type=str, default="/home/tuo28237/projects/fl/data/")
        ap.add_argument("-d", "--device", type=str, default="cpu")
        ap.add_argument("-r", "--result_dir", type=str, default="./result")
        ap.add_argument("-v", "--verbosity", type=int,default=1)
        ap.add_argument("-n", "--simulation_num", type=int, default=1)
        # self.ap.add_argument("-f", "--progress_file", type=str, default="./progress.txt")

        return ap

    def __init__(self, config: Config = None) -> None:
        if config == None:
            self.configs = Config()
        else:
            self.configs = copy.deepcopy(config)

    def start(self):
        if self.configs.verbosity >= 2:
            print("Arguments: %s %d %d %d %d %d %f %f %s %s %s" % (
                self.configs.task_name, self.configs.g_epoch_num,
                self.configs.client_num, self.configs.l_data_num, 
                self.configs.l_epoch_num, self.configs.l_batch_size,
                self.configs.l_lr, self.configs.sigma, self.configs.datapath,
                self.configs.device, self.configs.result_dir
                ))

        procs: list[Process] = []
        for i in range(self.configs.simulation_num):
            self.configs.simulation_index = i
            proc = Process(target=single_simulation, args=(self.configs,))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def set_configs(self, config: Config):
        self.configs = copy.deepcopy(config)

    def get_configs_from_cml(self):
        ap = Simulator.__get_argument_parser()
        args = ap.parse_args()

        task_name: str = args.task_name # limited: FashionMNIST/SpeechCommand/
        # global parameters
        g_epoch_num: int = args.g_epoch_num
        # local parameters
        client_num: int = args.client_num
        l_data_num: int = args.l_data_num
        l_epoch_num: int = args.l_epoch_num
        l_batch_size: int = args.l_batch_size
        l_lr: float = args.l_lr
        # shared settings
        datapath: str = args.datapath
        device: torch.device = torch.device(args.device)
        result_dir: str = args.result_dir

        verbosity: int = args.verbosity
        simulation_num: int = args.simulation_num
        # self.progress_file: str = self.ap.progress_file

        self.configs = Config(task_name, g_epoch_num, client_num,
            l_data_num, l_epoch_num, l_batch_size, l_lr, datapath,
            device, result_dir, verbosity, simulation_num)

    def check_configs(self) -> bool:
        # device check
        if self.configs.device == torch.device("cuda"):
            real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.configs.device != real_device:
                raise "Error: cuda wanted but not equipped."
        elif self.configs.device != torch.device("cpu"):
            raise "Error: target device is not cpu nor cuda, unspported"

        if self.configs.task_name not in UniTask.supported_tasks:
            raise "Task not supported yet."
