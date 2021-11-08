
import random

from torch import Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, Subset
from torchvision import datasets, transforms

from utils.tasks import Config
from utils.data import dataset_split

dataset = datasets.FashionMNIST(
                root="./data/",
                train=True,
                download=False,
                transform=transforms.ToTensor(),
                )

# subset = Subset(dataset, index_list[5])
config = Config()
# config.sigma = 0
config.client_num = 6
subsets = dataset_split(dataset, config, 1)
lable_list = []
for (sample, lable) in subsets[4]:
    lable_list.append(lable)

# # print(lable_list)

sum = 0
for i in range(10):
    count = lable_list.count(i)
    print(count)
    sum += count

print(sum)