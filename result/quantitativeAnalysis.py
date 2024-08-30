import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((original_data - imputed_data)**2))





def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr



def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

if __name__ == '__main__':
    imputed_data_path = r"Usoskin_imputation.csv"
    original_data_path = r"Usoskin.csv"
    imputed_data = pd.read_csv(imputed_data_path, sep=',', index_col=0).values
    original_data = pd.read_csv(original_data_path, sep=',', index_col=0).values #header=None, index_col=False
    print(imputed_data.shape)
    print(original_data.shape)

    pccs = pearson_corr(imputed_data,original_data)
    print("pccs=",pccs)

    rmse = RMSE(imputed_data,original_data)
    print("rmse=",rmse)
    
    r_squared_example = calculate_r_squared(original_data, imputed_data)
    print("r^2=",r_squared_example)

