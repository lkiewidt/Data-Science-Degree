# Finding Donors with Charity ML

## Introduction
The Finding Donors with Charity ML is part of Udacity's Data Scientist Nanodegree. The goal is to identify individuals as donors for Charity ML based on their employment and social features.

In the project, a training-evaluation-tunig pipeline for is developed for several supvervised learning algorithms. 

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.3
* numpy = 1.16.4
* pandas = 0.24.2
* sklearn = 0.21.2
* matplotlib = 3.1.0

## Key Findings
* the Naive-Bayes classifier is fast to train, however, the predictions are only slighty better than a fully naive model which predicts all individuals as potential donors
* the Support Vector classifier (SVC) produces good precision and recall scores compared to the fully naive model, however, at the cost of significantly longer training and prediction times
* the Random Forest classifier yields precision and recall scores comparable to the ones from the SVC at low training and prediction times
    * tuning of hyper-parameters improved the prediction slightly
* from the 13 features, 5 contribute 60% to the variance in the dataset
    * the most relevant features for finding potential donors with high income relate to the ability to build up capital, eduction, relationship status, and age
