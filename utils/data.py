from torch.utils.data.dataset import Dataset, Subset

from utils.tasks import Config

def dataset_categorize(dataset: Dataset):
    targets = dataset.targets.tolist()
    targets_list = list(set(targets))
    # can be deleted, does not matter but more clear if kept
    targets_list.sort()

    indices_by_lable = [[] for target in targets_list]
    for i, target in enumerate(targets):
        category = targets_list.index(target)
        indices_by_lable[category].append(i)

    subsets = [Subset(dataset, indices) for indices in indices_by_lable]
    return subsets

def dataset_split(dataset: Dataset, config: Config, sigma: float=0.5, func_type: int=0) -> 'list[Dataset]':
    if func_type == 0:
        return dataset_split_0(dataset, config, sigma)
    if func_type == 1:
        return dataset_split_1(dataset, config, sigma)

def dataset_split_0(dataset: Dataset, config: Config, sigma: float=0.5) -> 'list[Dataset]':
    categorized_sets = dataset_categorize(dataset)
    



def dataset_split_1(dataset: Dataset, config: Config, sigma: float=0.5) -> 'list[Dataset]':
    pass