import numpy as np
import pandas as pd


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


    data_non_path = "..\data\mask40_Zeisel.csv"
    imputed_data_path = "..\data\imputation.csv"
    original_data_path = "..\data\\normalization_Zeisel.csv"

    data_non = pd.read_csv(data_non_path, sep=',', index_col=0).values
    imputed_data = pd.read_csv(imputed_data_path, sep=',', index_col=0).values
    original_data = pd.read_csv(original_data_path, sep=',', index_col=0).values


    # 计算PCCs
    pccs_non = pearson_corr(data_non,original_data)
    pccs = pearson_corr(imputed_data,original_data)

    # 计算RMSE
    rmse_non = RMSE(data_non,original_data)
    rmse = RMSE(imputed_data,original_data)

    # 计算R²
    r_squared_example_non = calculate_r_squared(data_non, imputed_data)
    r_squared_example = calculate_r_squared(original_data, imputed_data)

    print("===============")
    print("插补前pccs={:.3f}".format(pccs_non))
    print("插补前r^2={:.3f}".format(r_squared_example_non))
    print("插补前rmse={:.3f}*10e3".format(rmse_non*1000))
    print("===============")
    print("插补后pccs={:.3f}".format(pccs))
    print("插补后r^2={:.3f}".format(r_squared_example))
    print("插补后rmse={:.3f}*10e3".format(rmse*1000))