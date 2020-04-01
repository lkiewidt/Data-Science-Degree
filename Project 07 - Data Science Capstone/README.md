elderly# P-07: Data Science Capstone Project - Identifying customers with Arvato

## Introduction
The 'Data Science Capstone' project is the final assignment of of Udacity's Data Scientist Nanodegree. The goal is to identify potential customer segments for a mail-based sales company from demographic data of individuals living in Germany. The data was provided by Arvato, a Bertelsmann division. In addition to customer segment identification, a supervised machine learning classifier was developed to predict customers from their demographic features.

## Approach
In a first step, a dataset with more than 890,000 entries (rows) and more than 350 demographic features (columns) was analyzed, and rows and columns with high fractions of missing data were removed. Afterwards, new features were engineered from several mixed features in the original dataset. To reduce the feature space, highly correlated features (Pearson correlation coefficient higher/lower 0.8) were dropped as well. After cleaning, a Principal Component Analysis (PCA) was conducted to further reduce the number of features. The analysis of the contributions of individual features to the principal components allowed the identification of socio-economic groups (e.g. young urban populations, traditionally-minded elderly, or assertive male investors). Finally, k-Means was used to cluster the data.

In a second step, the trained clustering model was used to cluster data from a second dataset with existing customers of the mail-based sales company. Pair-wise comparison of the proportion of each cluster in the population and the customer dataset revealed over- and under-represented clusters in the customer dataset. The over-represented clusters were finally identified as potential customers segments for the sales company.

Afterwards, a machine learning classifier based on GradientBoosting was trained to predict customers from their demographic features. Therefore, a labeled dataset was provided. A major challenge in the development of the prediction model was the highly imbalanced class distribution: only 1.2% of individuals in the training dataset responded to an advertisement campaign. Therefore, Receiver-Operator-Curves (ROC) were used to assess the performance of the classifier. After feature selection and hyper-parameter optimization, the trained classifier obtained a ROC-AUC score of 0.77 and 0.78 on an unseen part of the provided data and in the [**Kaggle competition**](https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard), respectively.

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.5
* numpy = 1.17.4
* pandas = 0.25.3
* sklearn = 0.21.3
* matplotlib = 3.1.0

## Key Findings
* comparison of the general population and customer datasets revealed individuals from wealthy households as well as elderly, traditionally-minded persons as most promising customer segment for the sale company
  * PCA and clustering are powerful tools to analyze large datasets and to create insights into complex socio-economic questions
* although PCA was essential to identify the customer segments, the principal components (PCs) were not able to successfully predict customers from the provided training data with high confidence
* for highly imbalanced classification problem, feature selection and engineering is key to build confidence classification models; therefore, significant domain knowledge is necessary
