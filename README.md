# Data Science Portfolio

## Introduction
During my *Data Science Nanodegree* at [Udacity](https://eu.udacity.com), I completed the following projects:


## Project 01: Finding donors
**Abstract:**  The goal of the 'Finding donors' project is to identify potential donors for a fictitious charity organization based on their social and professional features. Therefore, multiple supervised machine learning algorithms were trained, tested, and compared to each other. The hyper-parameters of the most promising algorithm (Random Forest) were further optimized to improve prediction accuracy and F-score. Finally, the most relevant features were analyzed to create insights insights into the identification of potential donors.

**Keywords:** supervised learning, classification, Naive Bayes, Support Vector Machine, Random Forest, parameter tuning

[**Link to jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2001%20-%20Finding%20Donors/finding_donors.ipynb)


## Project 02: Image classifier
**Abstract:**  In the 'Image classifier' project, a deep neural network was build and trained to identify 102 different flower species from images. The neural network consisted of a pre-trained convolutional network (VGG16) and a fully connected classifier with a single hidden layer. After training over 5 epochs, the network reached an accuracy of more than 70%. Finally, two functions, `train.py` and `predict.py` were implemented deploy the model, e.g., to the command line or to an online application.

**Keywords:** deep learning, neural networks, image classification, PyTorch

[**Link to jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2002%20-%20Image%20Classifier/Image%20Classifier%20Project.ipynb)


## Project 03: Identifying customer segments
**Abstract:** In the 'Identifying customer segments' project, demographic data of individuals living in Germany and customer data of a mail-based sales company were clustered and compared to identify potential customer segments for the sales company. The data was provided by Arvato, a Bertelsmann division. Prior to a Principal Component Analysis (PCA) and clustering with k-Means, the data was analyzed in detail. Missing values were dropped from the dataset and new features were engineered from existing mixed-type, multi-categorical features. The comparison of the share of the identified clusters in the demographic and the customer dataset revealed elderly, traditionally-minded persons as promising potential customers for the sales company.

**Keywords:** unsupervised learning, Principal Component Analysis (PCA), clustering, k-Means, data wrangling, feature engineering

[**Link to jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2003%20-%20Indentifying%20Customer%20Segments/Identify_Customer_Segments.ipynb)


## Project 04: Date Science blog post
**Abstract:** In this project, data from [Nature's 2017 Graduate Survey](https://www.nature.com/nature/journal/v550/n7677/full/nj7677-549a.html) was analyzed to find out more about the motivation of students to pursue a PhD, the skills they learn, and their satisfaction with their PhD program. Further, the possibility to predict the satisfaction score based on prior features that are known before the PhD students start, such as personal motivation and the field they work in, was explored. Prior to the analysis, the data was cleaned and useful features were selected and properly encoded. The analysis showed that most students pursue a PhD to follow an academic career although they are aware that faculty positions at universities are scarce. Nevertheless, PhD students obtain a broad range of skills from collecting and analyzing data, over communicating their findings, to managing people and projects. In summary the data shows that the decision to do a PhD is a highly individual one that probably depends on many personal aspects that are difficult to capture and extract from data.

**Keywords:** data science, data wrangling, feature selection, visualization, data-based communication

[**Link to jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2004%20-%20Data%20Science%20Blog%20Post/dataScience_blogPost_NaturePhDSurvery.ipynb)

[**Link to blog post**](https://medium.com/@kiewidt/to-phd-or-not-to-phd-4312cdb862c5)


## Project 05: Disaster Response Pipeline
**Abstract:** In the Disaster Response project, text messages send during disaster situations were analyzed and classified into several categories (e.g. fire, flood, storm, etc.) to allow quick and effective deployment of support. Therefore, ETL (extract-transform-load) and ML (machine learning) pipelines were implemented to enable automated and straightforward processing of datasets and training and testing of the classifier. The trained classifier was deployed in a web-based app to classify incoming messages from a web-browser.

**Keywords:** natural language processing (NLP), ETL pipelines, ML pipelines, web development

[**Link to ETL pipeline jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2005%20-%20Disaster%20Response%20Pipeline/data/ETL%20Pipeline%20Preparation.ipynb)

[**Link to ML pipeline jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2005%20-%20Disaster%20Response%20Pipeline/models/ML%20Pipeline%20Preparation.ipynb)


## Project 06: Recommendation Systems
**Abstract:** In the Recommendation Systems project, several recommendation techniques were implemented for real data from IBM's Watson Studio platform. The goal was to recommend articles on Watson Studio to new and existing users. For existing users, user-user-based collaborative filtering was applied to make recommendations based on the similarity to other users, total article interactions, and total article views. For new users, for whom the similarity to other users could not be determined due to missing viewing history (cold-start problem), a ranked-based recommendation technique was applied, which recommends the most-viewed articles. Finally, matrix factorization (singular value decomposition, SVD) was implemented to identify latent features and predict user-item interactions.

**Keywords:** collaborative filtering, rank-based recommendations, user-based recommendations, matrix factorization, singular value decomposition (SVD)

[**Link to jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2006%20-%20Recommendation%20Engine%20IBM%20Studio/Recommendations_with_IBM.ipynb)


## Project 07: Data Science Capstone Project - Identifying customers
**Abstract:** In the 'Data Science Capstone Project', I choose to extent the customer identification from Project 03. In addition to the identification of demographic clusters that are overrepresented in the customer base of the fictitious mail-order company by Principal Component Analysis (PCA) and clustering with k-Means, a supervised machine learning model (based on GradientBoosting) was trained to predict potential customers from their demographic features. Further, more than 250 additional demographic features were included in the analysis. Therefore, more focus was put on the correlation between several features in the datasets to select relevant features for the cluster analysis and the prediction. As in Project 03, the clustering analysis showed that individuals from wealthy households and elderly persons are the customer base of the mail-order company. A major challenge in training the classifier was the highly imbalanced training data: only 1.2% of individuals in the training dataset became a customer after they were addressed in an advertisement campaign. Therefore, the performance of the classifier was evaluated through Receiver-Operator-Curves (ROC). After feature selection and hyper-parameter optimization, the trained classifier obtained a ROC-AUC score of 0.77 and 0.78 on an unseen part of the provided data and in the [**Kaggle competition**](https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard), respectively.

**Keywords:** unsupervised learning, Principal Component Analysis (PCA), clustering, k-Means, supervised learning, imbalanced classification, ROC-AUC

[**Link to jupyter notebook**](https://github.com/lkiewidt/Data-Science-Degree/blob/master/Project%2007%20-%20Data%20Science%20Capstone/Arvato%20Project%20Workbook.ipynb)

[**Link to blog post**](https://medium.com/@kiewidt/whos-next-in-the-line-adb147404dc9)
