# Identifying customer segments with Arvato

## Introduction
The Identifying customer segments is part of Udacity's Data Scientist Nanodegree. The goal is to identify potential customer segments for a mail-based sales company from demographic data of individuals living in Germany.

In addition to data cleaning (removing or imputing missing data points), new features are created from existing multi-categorical features. From the 81 features, 20 principal features are created using principal component anaylsis (PCA). Afterwards, k-Means is used to cluster the data points. A comparison of the obtained clusters with 

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.3
* numpy = 1.16.4
* pandas = 0.24.2
* sklearn = 0.21.2
* matplotlib = 3.1.0

## Key Findings
* data cleaning and feature selection and engineering are important when working with large datasets that are partially compiled from public sources; it takes up a major fraction to the total work to cluster data
* PCA is a powerful tool to analyze large datasets and to obtain typical personas for interesting clusters
* comparison of the demographic and customer datasets allows identification of over- and under-represented clusters in the customer dataset and thus finding potential customer segments

