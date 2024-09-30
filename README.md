# BiAEImpute
BiAEImpute: Enhancing Dropout Imputation in Single-Cell RNA-seq with Bidirectional Autoencoders
# Overview
We proposed a bidirectional autoencoder-based model (BiAEImpute) for dropout imputation in scRNA-seq dataset. This model employs row-wise autoencoders and column-wise autoencoders to respectively learn cellular and genetic features during the training phase. The synergistic integration of these learned features is then utilized for the imputation of missing values, enhancing the robustness and accuracy of the imputation process.
![figure_1](https://github.com/user-attachments/assets/97b89162-41a3-4b22-8da7-0504b9f7e641)

# Datasets
**Zeisel**: The Zeisel dataset contains 19,972 genes and 3,005 cells, representing seven cell types from cerebral cortex tissue (interneurons, pyramidal SS, pyramidal CA1, oligodendrocytes, microglia, endothelial-mural, and astrocytes-ependymal), and is available in the NCBI repository under the GEO accession number GSE60361.<br>
**Romanov**: The Romanov dataset consists of 24341 genes and 2881 cells across seven cell types from hypothalamus tissue (oligodendrocytes, astrocytes, ependymal, microglia, vsm, endothelial, neurons), and is available in the NCBI repository under the GEO accession number GSE74672.<br>
**Usoskin**: The Usoskin dataset has 25334 genes and 622 cells, representing four cell types from lumbar dorsal root ganglion tissue (NF, NP, PEP, TH), and is available in the NCBI repository under the GEO accession number GSE59739.<br>
**Klein**: The Klein dataset is a longitudinal dataset containing 24175 genes and 2717 cells from embryonic stem cells, sampled at four time points (day 0, day 2, day 4, day 7), and is available in the NCBI repository under the GEO accession number GSE65525.
# Getting Started
BiAEImpute can be used via the command line. The following instructions will guide you through quickly setting it up and running on your local machine.
## Usage
### Installing
You can clone this directory to install the latest GitHub version:
```bash
git clone https://github.com/LiuXinyuan6/BiAEImpute.git
```
### Usage
#### 1.stage of training
```bash
python ./train.py --datasets Zeisel.csv --mask_ratio 0.4 --normalization True --eps 500
```
```--datasets``` is the dataset to be imputed, where rows are genes and columns are cells.
```--mask_ratio``` is the simulated missing probability, ranging from 0 to 1.
```--normalization``` indicates whether normalization is required, with a default value of "True."
```--eps``` is the number of training epochs, with a default value of "500."
#### 2.stage of inferring
```bash
python ./infer.py --datasets Zeisel.csv --mask_ratio 0.4
```
### Verify the experimental results of the Zeisel dataset with 40% dropout in this paper.
```python
    # Verify the experimental results of PCC、R²、RMSE
    data_non_path = "mask40_Zeisel.csv"
    imputed_data_path = "imputation.csv"
    original_data_path = "normalization_Zeisel.csv"

    data_non = pd.read_csv(data_non_path, sep=',', index_col=0).values
    imputed_data = pd.read_csv(imputed_data_path, sep=',', index_col=0).values
    original_data = pd.read_csv(original_data_path, sep=',', index_col=0).values

    # PCCs
    pccs_non = pearson_corr(data_non,original_data)
    pccs = pearson_corr(imputed_data,original_data)

    # RMSE
    rmse_non = RMSE(data_non,original_data)
    rmse = RMSE(imputed_data,original_data)

    # R²
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
```
```python
    # print
    ===============
    插补前pccs=0.771
    插补前r^2=0.665
    插补前rmse=6.549*10e3
    ===============
    插补后pccs=0.918
    插补后r^2=0.841
    插补后rmse=4.086*10e3
```
```python
    # 插补前的
    clusterResults_non = pd.read_csv("clustering_non-Imputed.csv",index_col=0)
    clusterResults_non = clusterResults_non.values.squeeze()
    # 插补后的
    clusterResults = pd.read_csv("clustering_imputation.csv",index_col=0)
    clusterResults = clusterResults.values.squeeze()
    # 真实标签
    labels = pd.read_csv("labels.csv",index_col=0)
    labels = labels.values.squeeze()

    ari_non = adjusted_rand_score(labels,clusterResults_non)
    nmi_non = normalized_mutual_info_score(labels,clusterResults_non)
    purity_non = getPurityScore(labels,clusterResults_non)

    ari = adjusted_rand_score(labels,clusterResults)
    nmi = normalized_mutual_info_score(labels,clusterResults)
    purity = getPurityScore(labels,clusterResults)

    print("插补前ari={:.3f}".format(ari_non))
    print("插补前nmi={:.3f}".format(nmi_non))
    print("插补前purity={:.3f}".format(purity_non))
    print("===============")
    print("插补后ari={:.3f}".format(ari))
    print("插补后nmi={:.3f}".format(nmi))
    print("插补后purity={:.3f}".format(purity))
```
```python
    # print
    插补前ari=0.473
    插补前nmi=0.528
    插补前purity=0.688
    ===============
    插补后ari=0.860
    插补后nmi=0.829
    插补后purity=0.934
```
```python
    real_times = pd.read_csv(r"..\data\traj_time_imputation.csv")
    real_times = real_times.values.flatten()
    inferred_times = pd.read_csv(r"..\data\traj_time_Klein.csv")
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
```
```python
    #print
    k_tau = 0.867
```



