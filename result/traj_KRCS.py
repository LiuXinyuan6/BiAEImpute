from scipy.stats import kendalltau
import pandas as pd


real_times = pd.read_csv(r"traj_time_Klein_imputation.csv")
real_times = real_times.values.flatten()
inferred_times = pd.read_csv(r"traj_time_Klein.csv")
inferred_times = inferred_times.values.flatten()




dat1 = real_times
dat2 = inferred_times
c = 0
d = 0
for i in range(len(dat1)):
    for j in range(i + 1, len(dat1)):
        if (dat1[i] - dat1[j]) * (dat2[i] - dat2[j]) > 0:
            c = c + 1
        else:
            d = d + 1
k_tau = (c - d) * 2 / len(dat1) / (len(dat1) - 1)

print('k_tau = {0}'.format(k_tau))
