
from argparse import ArgumentParser, Namespace
from multiprocessing import Process, Queue, set_start_method


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset, random_split

from utils.client import Client
from utils.server import Server
from utils.funcs import Config

def __single_simulation(
    configs: Config
    ):
    simulator = __SingleSimulator(configs)
    simulator.start()

class __SingleSimulator:
    def __init__(self, configs: Config) -> None:
        self.configs = configs

        self.server: Server = None
        self.clients: list[Client] = []

    def start(self):
        self.configure_clients()

        result_file = self.configs.result_dir + \
            "/result" + str(self.configs.simulation_num)
        f = open(result_file, "a")
        args = "{:12} {:11} {:10} {:10} {:11} {:12} {:4}".format(
            self.configs.task_name, self.configs.g_epoch_num, 
            self.configs.client_num, self.configs.l_data_num, 
            self.configs.l_epoch_num, self.configs.l_batch_size, 
            self.configs.l_lr)
        f.write("TASK          G_EPOCH_NUM CLIENT_NUM L_DATA_NUM " + 
            "L_EPOCH_NUM L_BATCH_SIZE L_LR\n" + args + "\n")
        f.flush()

        for i in range(self.configs.g_epoch_num):
            self.server.distribute_model()
            for client in self.clients:
                client.train_model()
            self.server.aggregate_model()

            if i % 10 == 9:
                g_accu = self.server.test_model()
                f.write("{:.2f} ".format(g_accu))
                f.flush()

    def configure_clients(self):
        
        clients = [Client]

    def get_partitioned_datasets(self,
        task: str,
        client_num: int,
        data_num: int,
        batch_size: int,
        data_path: str) \
        -> 'list[Subset]':

        if task == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(
                root=data_path,
                train=True,
                download=True,
                transform=ToTensor(),
                )
        elif task == "SpeechCommand":
            train_dataset = SubsetSC("training", data_path)
        elif task == "AG_NEWS":
            train_iter = AG_NEWS(split="train")
            train_dataset = to_map_style_dataset(train_iter)

        dataset_size = len(train_dataset)
        # subset division
        if data_num * client_num > dataset_size:
            raise "No enough data!"
        data_num_total = data_num*client_num
        subset = random_split(train_dataset, [data_num_total, len(train_dataset)-data_num_total])[0]
        subset_lens = [ data_num for j in range(client_num) ]
        subsets = random_split(subset, subset_lens)
        
        return subsets

    def get_test_dataset(self, task: str, data_path: str) -> Subset:
        if task == "FashionMNIST":
            test_dataset = datasets.FashionMNIST(
                root=data_path,
                train=False,
                download=True,
                transform=ToTensor()
                )
        elif task == "SpeechCommand":
            test_dataset = SubsetSC("testing", data_path)
        elif task == "AG_NEWS":
            test_iter = AG_NEWS(split="test")
            test_dataset = to_map_style_dataset(test_iter)

        return test_dataset

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
        ap.add_argument("-p", "--datapath", type=str, default="/home/tuo28237/projects/fledge/data/")
        ap.add_argument("-d", "--device", type=str, default="cpu")
        ap.add_argument("-r", "--result_dir", type=str, default="./result.txt")
        ap.add_argument("-v", "--verbosity", type=int,default=1)
        ap.add_argument("-n", "--simulation_num", type=int, default=1)
        # self.ap.add_argument("-f", "--progress_file", type=str, default="./progress.txt")

        return ap

    def __init__(self) -> None:
        self.supported_tasks = ["FashionMNIST", "SpeechCommand", "AG_NEWS"]
        self.configs: Config = None
        
        # self.parse_args()

    def start(self):
        if self.configs.verbosity >= 2:
            print("Arguments: %s %d %d %d %d %d %f %s %s %s" % (
                self.configs.task_name, self.configs.g_epoch_num,
                self.configs.client_num, self.configs.l_data_num, 
                self.configs.l_epoch_num, self.configs.l_batch_size,
                self.configs.l_lr, self.configs.datapath,
                self.configs.device, self.configs.result_dir
                ))

        set_start_method("spawn")
        procs: list[Process] = []
        for i in range(self.configs.simulation_num):
            self.configs.simulation_index = i
            proc = Process(
                    target=__single_simulation,
                    args=(self.configs, ))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()

    def parse_args(self):
        ap = self.__get_argument_parser()

        task_name: str = ap.task_name # limited: FashionMNIST/SpeechCommand/
        # global parameters
        g_epoch_num: int = ap.g_epoch_num
        # local parameters
        client_num: int = ap.client_num
        l_data_num: int = ap.l_data_num
        l_epoch_num: int = ap.l_epoch_num
        l_batch_size: int = ap.l_batch_size
        l_lr: float = ap.l_lr
        # shared settings
        datapath: str = ap.datapath
        device: torch.device = torch.device(ap.device)
        result_dir: str = ap.result_dir

        verbosity: int = ap.verbosity
        simulation_num: int = ap.simulation_num
        # self.progress_file: str = self.ap.progress_file


        self.check_device()
        self.configs = Config(task_name, g_epoch_num, client_num,
            l_data_num, l_epoch_num, l_batch_size, l_lr, datapath,
            device, result_dir, verbosity, simulation_num)

    def check_device(self) -> bool:
        if self.configs.device == torch.device("cpu"):
            return True
        elif self.configs.device == torch.device("cuda"):
            real_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.configs.device == real_device:
                return True
            else:
                raise "Error: cuda wanted but not equipped."
        else:
            raise "Error: target device is not cpu nor cuda, unspported"
