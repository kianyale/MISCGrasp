import random
import time

import numpy as np


def dummy_collate_fn(data_list):
    return data_list[0]


def simple_collate_fn(data_list):
    ks = data_list[0].keys()
    outputs = {k: [] for k in ks}
    for k in ks:
        for data in data_list:
            outputs[k].append(data[k])
        outputs[k] = torch.stack(outputs[k], 0)
    return outputs


def custom_collate_fn(data_dict):
    def collate(items):
        if isinstance(items[0], torch.Tensor):
            return torch.stack(items)
        elif isinstance(items[0], dict):
            result = {}
            for key in items[0]:
                if key == 'gripper' or key == 'gripper_param':
                    result[key] = torch.from_numpy(items[0][key])  # 直接返回第一个值
                elif key == 'gripper_type':
                    result[key] = items[0][key]
                else:
                    result[key] = collate([item[key] for item in items])
            return result
        elif isinstance(items[0], list):
            return [collate(subitems) for subitems in zip(*items)]
        else:
            np_items = np.array(items)
            return torch.from_numpy(np_items)

    return collate(data_dict)


import torch


def efficient_collate_fn(batch):
    """
    Optimized collate function for NewSepSymmGripperAwareDataset.
    Combines individual data samples into a batch with better performance.
    """
    # Assume batch[0] provides the structure of the data
    batch_size = len(batch)
    sample = batch[0]

    # Preallocate tensors based on the sample
    n_grasps = len(sample["grasp_info"]["pos"])
    volume_res_scn = sample["geo_info"]["scene"].shape[-1]
    volume_res_grp = sample["geo_info"]["gripper"].shape[-1]

    grasp_info = {
        "pos": torch.zeros((batch_size, n_grasps, 3), dtype=torch.float32),
        "index": torch.zeros((batch_size, n_grasps, 3), dtype=torch.long),
        "label": torch.zeros((batch_size, n_grasps), dtype=torch.float32),
        "rotation": torch.zeros((batch_size, n_grasps, 2, 4), dtype=torch.float32),
        "width": torch.zeros((batch_size, n_grasps), dtype=torch.float32),
        "gripper_type": torch.zeros((batch_size,), dtype=torch.long),
    }

    geo_info = {
        "scene": torch.zeros((batch_size, 1, volume_res_scn, volume_res_scn, volume_res_scn), dtype=torch.float32),
        "gripper": torch.zeros((batch_size, 2, volume_res_grp, volume_res_grp, volume_res_grp), dtype=torch.float32),
    }

    # Populate the preallocated tensors
    for i, data in enumerate(batch):
        grasp_info["pos"][i] = torch.tensor(data["grasp_info"]["pos"], dtype=torch.float32)
        grasp_info["index"][i] = torch.tensor(data["grasp_info"]["index"], dtype=torch.long)
        grasp_info["label"][i] = torch.tensor(data["grasp_info"]["label"], dtype=torch.float32)
        grasp_info["rotation"][i] = torch.tensor(data["grasp_info"]["rotation"], dtype=torch.float32)
        grasp_info["width"][i] = torch.tensor(data["grasp_info"]["width"], dtype=torch.float32)
        grasp_info["gripper_type"][i] = torch.tensor(data["grasp_info"]["gripper_type"], dtype=torch.long)

        geo_info["scene"][i] = torch.tensor(data["geo_info"]["scene"], dtype=torch.float32)
        geo_info["gripper"][i] = torch.tensor(data["geo_info"]["gripper"], dtype=torch.float32)

    # Combine the results
    batched_data = {
        "grasp_info": grasp_info,
        "geo_info": geo_info,
    }

    return batched_data


def efficient_collate_fn2(batch):
    """
    Optimized collate function for NewSepSymmGripperAwareDataset.
    Combines individual data samples into a batch with better performance.
    """
    # Assume batch[0] provides the structure of the data
    batch_size = len(batch)
    sample = batch[0]

    # Preallocate tensors based on the sample
    n_grasps = len(sample["grasp_info"]["pos"])
    volume_res_scn = sample["geo_info"]["scene"].shape[-1]

    grasp_info = {
        "pos": torch.zeros((batch_size, n_grasps, 3), dtype=torch.float32),
        "index": torch.zeros((batch_size, n_grasps, 3), dtype=torch.long),
        "label": torch.zeros((batch_size, n_grasps), dtype=torch.float32),
        "rotation": torch.zeros((batch_size, n_grasps, 2, 4), dtype=torch.float32),
        "width": torch.zeros((batch_size, n_grasps), dtype=torch.float32),
    }

    geo_info = {
        "scene": torch.zeros((batch_size, 1, volume_res_scn, volume_res_scn, volume_res_scn), dtype=torch.float32),
    }

    # Populate the preallocated tensors
    for i, data in enumerate(batch):
        grasp_info["pos"][i] = torch.tensor(data["grasp_info"]["pos"], dtype=torch.float32)
        grasp_info["index"][i] = torch.tensor(data["grasp_info"]["index"], dtype=torch.long)
        grasp_info["label"][i] = torch.tensor(data["grasp_info"]["label"], dtype=torch.float32)
        grasp_info["rotation"][i] = torch.tensor(data["grasp_info"]["rotation"], dtype=torch.float32)
        grasp_info["width"][i] = torch.tensor(data["grasp_info"]["width"], dtype=torch.float32)

        geo_info["scene"][i] = torch.tensor(data["geo_info"]["scene"], dtype=torch.float32)

    # Combine the results
    batched_data = {
        "grasp_info": grasp_info,
        "geo_info": geo_info,
    }

    return batched_data


def set_seed(index, is_train):
    if is_train:
        np.random.seed((index + int(time.time())) % (2 ** 16))
        random.seed((index + int(time.time())) % (2 ** 16) + 1)
        torch.random.manual_seed((index + int(time.time())) % (2 ** 16) + 1)
        torch.manual_seed((index + int(time.time())) % (2 ** 16) + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed((index + int(time.time())) % (2 ** 16) + 1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)
        torch.manual_seed(index % (2 ** 16) + 1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(index % (2 ** 16) + 1)

