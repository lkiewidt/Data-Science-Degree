# P01: Finding Donors with Charity ML

## Introduction
The 'Finding Donors with Charity ML' project is part of Udacity's Data Scientist Nanodegree. The goal is to identify individuals as potential donors for Charity ML, a fictious charity organization, based on their employment and social features. An individual income of at least 50,000 USD serves as label to identify a potential donor. The data originates from the US cencus. In the project, a training-evaluation pipeline for different supervised machine algorithms is developed. 

## Approach
After initial data exploration, feature encoding, and feature scaling, a training-evaluation pipeline for a first test of different machine learning algorithms was implemented. A Naive-Bayes classifier, a Support Vector classifier, and a Random Forest classifier were chosen. Training and testing accuracy and F-score, as well as training and prediction time were evaluated. Afterwards, the hyper-parameters of the most promising algorithm were optimized to further improve prediction accuracy and F-score.

## Python Libraries
The following Python libraries were use in the project:
* python = 3.7.3
* numpy = 1.16.4
* pandas = 0.24.2
* sklearn = 0.21.2
* matplotlib = 3.1.0

## Key Findings
* the Naive-Bayes classifier is fast to train, however, the predictions are only slighty better than a fully naive model (24% accuracy) which predicts all individuals as potential donors
* the Support Vector classifier (SVC) produces good accuracy (80%) and F-scores (60%) compared to the fully naive model, however, at the cost of significantly longer training and prediction times
* the Random Forest classifier yields accuracy and F-scores comparable to the ones of the SVC at low training and prediction times
    * tuning of hyper-parameters improved the prediction slightly (F-score from 68% to 73%)
* from the 13 features, 5 contribute 60% to the variance in the dataset
    * the most relevant features for finding potential donors with high income relate to the ability to build up capital, eduction, relationship status, and age
