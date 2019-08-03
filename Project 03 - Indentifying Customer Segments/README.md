# P-03: Identifying customer segments with Arvato

## Introduction
The 'Identifying customer segments' project is part of Udacity's Data Scientist Nanodegree. The goal is to identify potential customer segments for a mail-based sales company from demographic data of individuals living in Germany. The data was provided by Arvato, a Bertelsmann division.

## Approach
In a first step, a dataset with more than 890,000 entries (rows) and 85 features (columns) was analyzed and rows and columns with high fractions of missing data were removed. Afterwards, new features were engineered from several mixed features in the original dataset. After cleaning, a Principal Component Analysis (PCA) was conducted to reduce the number of features. The analysis of the contributions of individual features to the principal components allowed the identification of socio-economic groups (e.g. young urban populations, traditionally-minded elderly, or assertive male investors). Finally, k-Means was used to cluster the data.

In a second step, the trained clustering model was used to cluster data from a second dataset with existing customers of the mail-based sales company. Pair-wise comparison of the share of each cluster in the demographic and the customer dataset reveiled over- and under-represented clusters in the customer dataset. The over-represented clusters were finally identified as potential customers segements for the sales company.

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.3
* numpy = 1.16.4
* pandas = 0.24.2
* sklearn = 0.21.2
* matplotlib = 3.1.0

## Key Findings
* data cleaning and feature engineering and selection was crucial to identify customer segments reliably
    * approx. 20% of the rows from the demographic dataset were dropped due to a high fraction of missing values
    * after cleaning more than 600,000 entries remained in the demographic dataset for analysis 
    * from the the approximately 190,000 entries in the customer dataset, around 40% were removed due to a high fraction of missing values
* comparison of the demographic and customer datasets reveiled eldery, traditionally-minded persons as most promising customer segment for the sale company
* PCA and clustering are powerful tools to analyze large datasets and to create insights into complex socio-economic questions
