
import random

from torch.utils.data.dataset import Dataset, Subset

from utils.tasks import Config

def dataset_categorize(dataset: Dataset) -> 'list[int]':
    targets = dataset.targets.tolist()
    targets_list = list(set(targets))
    # can be deleted, does not matter but more clear if kept
    targets_list.sort()

    indices_by_lable = [[] for target in targets_list]
    for i, target in enumerate(targets):
        category = targets_list.index(target)
        indices_by_lable[category].append(i)

    # randomize
    for indices in indices_by_lable:
        random.shuffle(indices)

    # subsets = [Subset(dataset, indices) for indices in indices_by_lable]
    return indices_by_lable

def dataset_split(dataset: Dataset, config: Config, sigma: float=0.5, func_type: int=0) -> 'list[Dataset]':
    """
    Warnning:
    check config mannually that l_data_num * client_num <= dataset length
    For non-IID splitting, make sure the dataset can suffice the splitting
    """
    
    if func_type == 0:
        return dataset_split_0(dataset, config, sigma)
    if func_type == 1:
        return dataset_split_1(dataset, config, sigma)

def dataset_split_0(dataset: Dataset, config: Config, sigma: float=0.5) -> 'list[Dataset]':
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(config.client_num)]

    # fill the dominant type of data
    dominant_data_num = int(config.l_data_num*sigma)
    category_num = len(categorized_index_list)
    for i in range(len(indices_list)):
        cur_category = i % category_num
        indices_list[i] += categorized_index_list[cur_category][:dominant_data_num]
        categorized_index_list[cur_category] = categorized_index_list[cur_category][dominant_data_num:]

    # fill other types of data
    other_type_num = len(categorized_index_list) - 1
    other_data_num = int(config.l_data_num * (1 - sigma) / other_type_num)
    for i in range(len(indices_list)):
        dominant_category = i % category_num
        for j in range(len(categorized_index_list)):
            # not the dominant type
            if j != dominant_category:
                indices_list[i] += (categorized_index_list[j][:other_data_num])
                categorized_index_list[j] = categorized_index_list[j][other_data_num:]

    subsets = [ Subset(dataset, indices) for indices in indices_list ]
    return subsets




    



def dataset_split_1(dataset: Dataset, config: Config, sigma: float=0.5) -> 'list[Dataset]':
    pass