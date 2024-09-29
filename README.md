# BiAEImpute
BiAEImpute: Enhancing Dropout Imputation in Single-Cell RNA-seq with Bidirectional Autoencoders
# Overview
We proposed a bidirectional autoencoder-based model (BiAEImpute) for dropout imputation in scRNA-seq dataset. This model employs row-wise autoencoders and column-wise autoencoders to respectively learn cellular and genetic features during the training phase. The synergistic integration of these learned features is then utilized for the imputation of missing values, enhancing the robustness and accuracy of the imputation process.
![figure_1](https://github.com/user-attachments/assets/97b89162-41a3-4b22-8da7-0504b9f7e641)

# Datasets
**Zeisel**: The Zeisel dataset contains 19,972 genes and 3,005 cells, representing seven cell types from cerebral cortex tissue (interneurons, pyramidal SS, pyramidal CA1, oligodendrocytes, microglia, endothelial-mural, and astrocytes-ependymal), and is available in the NCBI repository under the GEO accession number GSE60361.<br>
**Romanov**: The Romanov dataset consists of 24341 genes and 2881 cells across seven cell types from hypothalamus tissue (oligodendrocytes, astrocytes, ependymal, microglia, vsm, endothelial, neurons), and is available in the NCBI repository under the GEO accession number GSE74672.
**Usoskin**: The Usoskin dataset has 25334 genes and 622 cells, representing four cell types from lumbar dorsal root ganglion tissue (NF, NP, PEP, TH), and is available in the NCBI repository under the GEO accession number GSE59739.
**Klein**: The Klein dataset is a longitudinal dataset containing 24175 genes and 2717 cells from embryonic stem cells, sampled at four time points (day 0, day 2, day 4, day 7), and is available in the NCBI repository under the GEO accession number GSE65525.
# Getting Started
BiAEImpute can be used either via the command line or as a Python package. The following instructions will guide you through quickly setting it up and running on your local machine.
## Usage
### Installing
You can clone this directory to install the latest GitHub version:
git clone https://github.com/LiuXinyuan6/BiAEImpute.git
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


