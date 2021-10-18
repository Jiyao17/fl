
from argparse import ArgumentParser
import copy
from multiprocessing import Process

import torch

from utils.client import Client
from utils.server import Server
from utils.tasks import UniTask, Config

def single_simulation(configs: Config):
    ssimulator = __SingleSimulator(configs)
    ssimulator.start()

class __SingleSimulator:
    def __init__(self, configs: Config) -> None:
        self.configs = configs
        # set server
        self.configs.reside = -1
        server_task = UniTask(self.configs).get_task()
        self.server = Server(server_task)
        # set clients
        self.clients: list[Client] = []
        for i in range(self.configs.client_num):
            new_configs = copy.deepcopy(self.configs)
            new_configs.reside = i
            # print("reside @ client %d in simu %d" % (new_configs.reside, new_configs.simulation_index))
            self.clients.append(Client(UniTask(new_configs).get_task()))

    def start(self):

        result_file = self.configs.result_dir + \
            "/result" + str(self.configs.simulation_index)
        f = open(result_file, "a")
        if self.configs.verbosity >= 3:
            print("writing to file: %s, simu num: %d" % (result_file, self.configs.simulation_index))
        args = "{:12} {:11} {:10} {:10} {:11} {:12} {:4}".format(
            self.configs.task_name, self.configs.g_epoch_num, 
            self.configs.client_num, self.configs.l_data_num, 
            self.configs.l_epoch_num, self.configs.l_batch_size, 
            self.configs.l_lr)
        f.write("TASK          G_EPOCH_NUM CLIENT_NUM L_DATA_NUM " + 
            "L_EPOCH_NUM L_BATCH_SIZE L_LR\n" + args + "\n")
        f.flush()

        for i in range(self.configs.g_epoch_num):
            self.server.distribute_model(self.clients)
            for client in self.clients:
                client.train_model()
            self.server.aggregate_model(self.clients)

            # record result
            if self.configs.verbosity >=3:
                g_accu, g_loss = self.server.test_model()
                print("accuracy %f in simulation %d at global epoch %d" %
                    (g_accu, self.configs.simulation_index, i))
            elif self.configs.verbosity >=2:
                if i % 10 == 9:
                    g_accu, g_loss = self.server.test_model()
                    print("accuracy %f, loss %f in simulation %d at global epoch %d" %
                        (g_accu, g_loss, self.configs.simulation_index, i))

            if i % 10 == 9:
                if self.configs.verbosity <=1:
                    g_accu, g_loss = self.server.test_model()
                f.write("{:.5f} ".format(g_accu))
                f.flush()

        # finished
        f.write("\n")
        f.close()


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
            print("Arguments: %s %d %d %d %d %d %f %s %s %s" % (
                self.configs.task_name, self.configs.g_epoch_num,
                self.configs.client_num, self.configs.l_data_num, 
                self.configs.l_epoch_num, self.configs.l_batch_size,
                self.configs.l_lr, self.configs.datapath,
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
