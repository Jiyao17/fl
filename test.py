
import random

from torch import Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, Subset
from torchvision import datasets, transforms

from utils.tasks import Config
from utils.data import dataset_split

# dataset = datasets.FashionMNIST(
#                 root="./data/",
#                 train=True,
#                 download=False,
#                 transform=transforms.ToTensor(),
#                 )

# subset = Subset(dataset, index_list[5])
# config = Config()
# config.sigma = 0
# subsets = dataset_split(dataset, config)
# lable_list = []
# for (sample, lable) in subsets[7]:
#     lable_list.append(lable)

# # print(lable_list)
# for i in range(10):
#     print(lable_list.count(i))

nums = [ [1], [2], [3] ]

for i in range(len(nums)):
    nums[i] = [0]

print(nums)