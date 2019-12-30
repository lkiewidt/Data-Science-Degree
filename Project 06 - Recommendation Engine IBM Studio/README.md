# P-06: Recommendation Systems at IBM Watson Studio

## Introduction
In the Recommendation Systems project, several recommendation techniques were implemented for real data from IBM's Watson Studio platform. The goal was to recommend articles on Watson Studio to new and existing users. For existing users, user-user-based collaborative filerting was applied to make recommendations based on the similartiy to other users, total article interactions, and total article views. For new users, for whom the similarity to other users could not be determined due to missing viewing history (cold-start problem), a rank-based recommendation technique was applied, which recommends the most-viewed articles. Finally, matrix factorization (singular value decomposition, SVD) was implemented to identify latent features and predict user-item interactions.

## Approach
Throughout the project, article views were used as principal metric to measure user-item interactions. For new users without viewing history, the most-viewed articles were used as recommendation. For existing users, most similar users were identified using the inner (dot) product of the user-item interactions (0 = did not view article, 1 = did view article). Afterwards, articles of the most similar users were individually sorted by total views and used as recommendations. Finally, latent features were identified to predict user-item interactions as recommendations using singular value decomposition (SVD). The performance of the SVD was evaluated by splitting the dataset into a training and testing dataset.

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.5
* numpy = 1.17.4
* pandas = 0.25.3
* matplotlib = 3.1.0
* seaborn = 0.9.0

## Key Findings
* recommendation systems require multiple techniques, e.g., rank-based, content-based, and user-based, to make reliable recommendations for existing and new users
* to assess the performance of recommendation systems, a metric that evaluates whether a user likes or dislikes an item is necessary
* for new users without interaction history, rank-based techniques and content-based techniques allow to make recommendations
* for existing users, user-user-based collaborative filtering allows to make useful recommendations