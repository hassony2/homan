from collections import defaultdict

import numpy as np
import torch


def collate(dict_list):
    collated_dict = defaultdict(list)
    for key, sample_vals in dict_list[0].items():
        for sample in dict_list:
            collated_dict[key].append(sample[key])
        if isinstance(sample_vals, np.ndarray):
            collated_dict[key] = np.stack(collated_dict[key])
        elif isinstance(sample_vals, torch.Tensor):
            collated_dict[key] = torch.stack(collated_dict[key])
    return dict(collated_dict)

