import random
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict


class MyDataset(Dataset):
    def __init__(self, path, num_item, num_groups):
        super(MyDataset, self).__init__()

        self.data = np.load(path + 'train.npy', allow_pickle=True)
        self.num_groups = num_groups


        self.all_set = set(range(num_item))


    def __getitem__(self, index):
        user, pos_item, genre, user_prior, _ = self.data[index]

        neg_item = np.random.choice(list(self.all_set.difference([pos_item])), 1)[0]

        ###change genre to one-hot,[0,0,0,0,0,6]
        genre_lable = eval(genre)
        genre_lable_onehot = np.zeros(self.num_groups, dtype=np.int64)
        genre_lable_onehot[genre_lable] = genre_lable

        user_prior = np.array(user_prior, dtype=np.float32)


        return [user, pos_item, neg_item, genre_lable_onehot, user_prior]

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
