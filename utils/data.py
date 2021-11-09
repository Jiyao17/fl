
import random

from torch.utils.data.dataset import Dataset, Subset

from utils.tasks import Config

TEST_TYPES = ["iid-range", "noniid-sigma", "noniid-sigma-group", "noniid-r", "noniid-r-group"]


def dataset_categorize(dataset: Dataset) -> 'list[list[int]]':
    """
    return value:
    list[i] = list[int] = all indices for category i
    """
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

def dataset_split(dataset: Dataset, config: Config) -> 'list[Dataset]':
    """
    Warnning:
    check config mannually that l_data_num * client_num <= dataset length
    For non-IID splitting, make sure the dataset can suffice the splitting
    """
    
    if config.test_type[:12] == "noniid-sigma":
        # non-iid spliting
        return dataset_split_sigma(dataset, config)
    if config.test_type == "iid-range":
        # iid spliting, 5000-7000
        return dataset_split_iid_range(dataset, config)
    if config.test_type[:8] == "noniid-r":
        return dataset_split_r(dataset, config)

def dataset_split_sigma(dataset: Dataset, config: Config) -> 'list[Dataset]':
    """
    return value:
    list[Dataset], list[i]: a dataset dominated by category (i % category_num)
    i_max = client_num
    """
    # each dataset is dominated by one class (occupy sigma*l_data_num)
    # when sigma=1/target_type_num, the datasets are IID
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(config.client_num)]

    # fill the dominant type of data
    sigma = config.sigma
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

def dataset_split_iid_range(dataset: Dataset, config: Config) -> 'list[Dataset]':

    random.seed()
    subsets: list[Subset] = [ None for i in range(config.client_num)]
    indices = [ i for i in range(len(dataset))]
    random.shuffle(indices)
    start_point = 0
    for i in range(config.client_num):
        data_num = random.randrange(5000, 7000)
        subsets[i] = Subset(dataset, indices[start_point : start_point + data_num])
        start_point += data_num

    return subsets

def dataset_split_r(dataset: Dataset, config: Config) -> 'list[Dataset]':
    """
    r = config.sigma
    return value:
    list[Dataset], list[i]: a dataset contains r categories
    i_max = client_num
    """
    # each dataset is dominated by one class (occupy sigma*l_data_num)
    # when sigma=1/target_type_num, the datasets are IID
    categorized_index_list = dataset_categorize(dataset)
    indices_list = [[] for i in range(config.client_num)]

    # fill the dominant type of data
    r = config.sigma
    dominant_data_num = int(config.l_data_num*r)
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
