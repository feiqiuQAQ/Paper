import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self, batch_data, num_features):
        self.data = batch_data
        self.keys = list(self.data.keys())  # 获取所有键
        self.num_features = num_features + 2

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # 确保索引在数据集范围内
        if idx >= len(self.keys):
            raise IndexError("Index out of range")
        
        key = self.keys[idx]  # 获取数据的键
        try:
            data = self.data[key]
            X = data[:, 1: self.num_features]
            y = data[:, self.num_features:]
            return (X, y)
        except KeyError as e:
            print(f"KeyError: {e} at index {idx} (key: {key})")
            raise e
        except Exception as e:
            print(f"Error: {e} at index {idx} (key: {key})")
            raise e
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# class MyDataSet(Dataset):
#     def __init__(self, batch_data):
#         self.data = batch_data
#         self.keys = list(self.data.keys())

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         if idx >= len(self.keys):
#             raise IndexError("Index out of range")
        
#         key = self.keys[idx]
#         try:
#             data = self.data[key]
#             X = torch.tensor(data[:, 1:13], dtype=torch.float32)
#             y = torch.tensor(data[:, 13:], dtype=torch.float32)
#             return (X, y)
#         except KeyError as e:
#             print(f"KeyError: {e} at index {idx} (key: {key})")
#             raise e
#         except Exception as e:
#             print(f"Error: {e} at index {idx} (key: {key})")
#             raise e

# def add_noise(data, noise_factor=0.01):
#     noisy_data = data + noise_factor * torch.randn_like(data)
#     return noisy_data

# def time_shift(data, shift_max=2):
#     shift = np.random.randint(-shift_max, shift_max)
#     shifted_data = torch.roll(data, shifts=shift, dims=0)
#     return shifted_data

# def random_deletion(data, delete_fraction=0.1):
#     data_len = len(data)
#     delete_num = int(data_len * delete_fraction)
#     delete_indices = np.random.choice(data_len, delete_num, replace=False)
#     data[delete_indices] = 0
#     return data

# class AugmentedDataset(MyDataSet):
#     def __init__(self, data, augmentations):
#         super().__init__(data)
#         self.augmentations = augmentations

#     def __getitem__(self, idx):
#         X, y = super().__getitem__(idx)
#         for augmentation in self.augmentations:
#             X = augmentation(X)
#         return X, y
