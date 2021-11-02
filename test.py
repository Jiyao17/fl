
from torch import Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, Subset
from torchvision import datasets, transforms

dataset = datasets.FashionMNIST(
                root="./data/",
                train=True,
                download=False,
                transform=transforms.ToTensor(),
                )

targets = dataset.targets.tolist()
targets_list = list(set(targets))
targets_list.sort()
print(targets_list)
indices_by_lable = [[] for target in targets_list]
for i, target in enumerate(targets):
    category = targets_list.index(target)
    indices_by_lable[category].append(i)

subsets = [Subset(dataset, indices) for indices in indices_by_lable]
# subset = Subset(dataset, indices_by_lable[8])
print(subsets[5][0][1])


# dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]