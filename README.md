# TCGA Pan-Cancer RNA-Seq Clustering

This project explores unsupervised clustering of gene expression data from The Cancer Genome Atlas (TCGA). The goal was to investigate how dimensionality reduction and clustering techniques perform on high-dimensional RNA-seq datasets.

## Dataset

The dataset contains gene expression measurements for:

- 801 tumor samples
- 20,531 genes
- ~16 million total datapoints

Each sample represents normalized RNA-seq counts across thousands of genes.

## Methodology

The analysis pipeline included several preprocessing and dimensionality reduction steps:

1. Log normalization of RNA-seq expression values
2. Selection of high-variance genes to reduce dimensionality
3. Principal Component Analysis (PCA) to reduce the feature space to 50 components

After preprocessing, several clustering algorithms were applied:

- K-Means
- Agglomerative Hierarchical Clustering
- DBSCAN

Cluster quality was evaluated using:

- Silhouette score
- Davies-Bouldin index

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## Purpose

This project was part of a data mining / machine learning exploration of large-scale biological datasets and demonstrates how traditional clustering approaches perform on high-dimensional genomics data.
