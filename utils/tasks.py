
class Task:
    def __init__(self,
        task_name: int,
        l_data_num: int,
        l_epoch_num: int,
        l_batch_size: int,
        l_lr: float,
        data_path: str,
        device: str
    ):
        self.task_name = task_name
        self.l_data_num = l_data_num
        self.l_epoch_num = l_epoch_num
        self.l_batch_size = l_batch_size
        self.l_lr = l_lr
        self.data_path = data_path
        self.device = device

        self.dataset = None
        self.model = None

    def train():
        pass

    def test():
        pass

class FahionMNIST(Task):
    def __init__(self, 
        task_name: int,
        l_data_num: int,
        l_epoch_num: int, 
        l_batch_size: int, 
        l_lr: float, 
        data_path: str, 
        device: str):

        super().__init__(
            task_name,
            l_data_num, l_epoch_num, l_batch_size, l_lr,
            data_path, device
            )