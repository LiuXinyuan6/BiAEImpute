import os
from os.path import join
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import *
import torch
from torch.utils.data import Dataset, DataLoader

config = Config()


def save_Identify_trueValue(identifyData,inferData):
    temp = identifyData.copy()
    for i in range(identifyData.shape[0]):
        for j in range(identifyData.shape[1]):
            if identifyData[i, j] == 0:
                temp[i, j] = inferData[i, j]

    pd.DataFrame(temp).to_csv(config.data_root + "imputation.csv")




class RowDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_rows, self.n_cols = data.shape # 3005 2000


    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):

        row_data = self.data[idx, :]
        return row_data, idx


class ColDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_rows, self.n_cols = data.shape  # 3005 2000

    def __len__(self):
        return self.n_cols

    def __getitem__(self, idx):

        col_data = self.data[:, idx]
        return col_data, idx




