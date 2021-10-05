
import torch


class Config:
    def __init__(self,
        task_name: str,
        g_epoch_num: int,
        client_num: int,
        l_data_num: int,
        l_epoch_num: int,
        l_batch_size: int,
        l_lr: int,
        datapath: int,
        device: int,
        result_dir: str,
        verbosity: int,
        simulation_num: int,
        reside: int=0,
        simulation_index: int=None,
        ) -> None:

        self.task_name: str = task_name
        # global parameters
        self.g_epoch_num: int = g_epoch_num
        # local parameters
        self.client_num: int = client_num
        self.l_data_num: int = l_data_num
        self.l_epoch_num: int = l_epoch_num
        self.l_batch_size: int = l_batch_size
        self.l_lr: float = l_lr
        # shared settings
        self.datapath: str = datapath
        self.device: torch.device = torch.device(device)
        self.result_dir: str = result_dir
        self.verbosity:int = verbosity
        # run multiple simulations in processes at one time
        self.simulation_num: int = simulation_num
        # task reside on server (-1) or client (0, 1, ..., client_num-1)
        self.reside:int = reside
        # for single simulators to know its index
        # so it can write results to its file
        self.simulation_index:int = simulation_index

