import random
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict


class MyDataset(Dataset):
    def __init__(self, path, num_item):
        super(MyDataset, self).__init__()
        data_path = path

        self.data = np.load(data_path + 'train.npy').squeeze()

        self.all_set = set(range(num_item))


    def __getitem__(self, index):
        user, pos_item = self.data[index]

        neg_item = np.random.choice(list(self.all_set.difference([pos_item])), 1)[0]

        return [user, pos_item, neg_item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    num_item = 100
    num_user = 1000
    dataset = MyDataset('./Data/', num_user, num_item)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for data in dataloader:
        user, pos_items, neg_items = data
        print(user)
