import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
from os.path import join

#from utils import preprocess

class SingleCell(Dataset):
    def __init__(self, data_root, dataset_name):
        self.data_root = data_root
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.X)

    def load_data(self,mask_ratio):
        self.X = pd.read_csv(self.data_root + self.dataset_name,index_col=0)
        if not mask_ratio==0:
            self.X = self.dropout(self.X,mask_ratio)
        self.X = self.X.values
        return self.X



    def __getitem__(self, i):
        return self.X[i]

    def preprocessData(self,data_matrix,note="normalization_"):#,data
        """
        Perform max-min normalization on the data matrix.
        :param data_matrix: A 2D numpy array where rows correspond to genes and columns to cells.
        :return: A 2D numpy array with normalized data.
        """
        # Calculate the minimum and maximum for each column (cell)
        min_vals = data_matrix.min(axis=0)
        max_vals = data_matrix.max(axis=0)
        # Perform max-min normalization
        normalized_data = (data_matrix - min_vals) / (max_vals - min_vals)
        # Replace NaNs that result from division by zero with zeros (case when max = min)
        normalized_data = np.nan_to_num(normalized_data)
        normalized_data = pd.DataFrame(normalized_data)
        normalized_data.to_csv(self.data_root + note + self.dataset_name)
        return normalized_data




    def dropout(self,X,mask_ratio):
        non_zero_positions = np.where(X.to_numpy() > 0)
        num_mask = int(len(non_zero_positions[0]) * mask_ratio)
        mask_indices = np.random.choice(range(len(non_zero_positions[0])), num_mask, replace=False)
        for idx in mask_indices:
            X.iat[non_zero_positions[0][idx], non_zero_positions[1][idx]] = 0
        masked_csv_path = self.data_root + f"mask{int(mask_ratio * 100)}_" + self.dataset_name
        X.to_csv(masked_csv_path)
        print(f"Data masked with {mask_ratio * 100}% zeros saved to {masked_csv_path}")
        return X
