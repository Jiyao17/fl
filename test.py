
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

# targets = dataset.targets.tolist()
# targets_list = list(set(targets))
# targets_list.sort()
# print(targets_list)
# indices_by_lable = [[] for target in targets_list]
# for i, target in enumerate(targets):
#     category = targets_list.index(target)
#     indices_by_lable[category].append(i)

# for indices in indices_by_lable:
#     random.shuffle(indices)

# config = Config()
# sigma = 0.5

# categorized_index_list = dataset_categorize(dataset)
# index_list = [[] for i in range(config.client_num)]

# # fill the dominant type of data
# dominant_data_num = int(config.l_data_num*sigma)
# category_num = len(categorized_index_list)
# for i in range(len(index_list)):
#     cur_category = i % category_num
#     index_list[i] += categorized_index_list[cur_category][:dominant_data_num]
#     categorized_index_list[cur_category] = categorized_index_list[cur_category][dominant_data_num:]


# # fill other types of data
# other_type_num = len(categorized_index_list) - 1
# other_data_num = int(config.l_data_num * (1 - sigma) / other_type_num)
# for i in range(len(index_list)):
#     dominant_category = i % category_num
#     for j in range(len(categorized_index_list)):
#         # not the dominant type
#         if j != dominant_category:
#             index_list[i] += (categorized_index_list[j][:other_data_num])
#             categorized_index_list[j] = categorized_index_list[j][other_data_num:]

# subset = Subset(dataset, index_list[5])

subsets = dataset_split(dataset, Config(), 0.8)
lable_list = []
for (sample, lable) in subsets[7]:
    lable_list.append(lable)

# print(lable_list)
for i in range(10):
    print(lable_list.count(i))