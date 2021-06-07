# Fraud Detection with Bank Transaction Data

# Introduction
Background: Fraud is a significant problem for any bank. Fraud can take many forms, which involves large amount of variables.

Data: This project is based on a credit card transaction dataset that contains information about 800,000 records and 29 features from a bank in Canada.

Objective: The goal is to build a predictive model to determine whether a given transaction will be fraudulent or not. 

# Methodology
Since the fraud detection is a binary classification problem, I applied several classic supervised classification machine learning models, such as Logistic regression, XGBoost, and LightGBM. The primary metric would be ROC_AUC.


# Data Preparation
Original dataset:

![ori_data](https://user-images.githubusercontent.com/64850893/103314535-68da1200-49f1-11eb-8c3f-4c007ba870f2.jpg)

Since this dataset is in line-delimited JSON format, I firstly transfromed it to a dataframe format.

![data](https://user-images.githubusercontent.com/64850893/103314331-c326a300-49f0-11eb-8381-0bae91605374.jpg)

# EDA
I utilized my domain knowlegdge in banking to select a few key features, and conducted descriptive analysis. Take the column "postEntryMode" for example:
Among all the fraudulent transactions, 

<img src="https://user-images.githubusercontent.com/64850893/103315082-94a9c780-49f2-11eb-854c-0aec766b7e63.jpg" width="600" height="300">

Among the not fraudulent transactions,

<img src="https://user-images.githubusercontent.com/64850893/103315273-14d02d00-49f3-11eb-9a83-596a7bde10fd.jpg" width="600" height="300">

From the above 2 plots, it seemed like "posEntryMode" had a influence on whether it's a fraud or not. So this feature would be added to my model.

## Feature engineering (3 cases)
Case 1: explanatory variables include 'cvvNotSame', 'amountOver', 'posEM_new', 'hour', 'transactionAmount', 'availableMoney', 'cardPresent'.

Case 2: explanatory variables include 'cvvNotSame', 'amountOver', 'posEM_new', 'hour', 'transactionAmount', 'availableMoney', 'cardPresent', 'merchantCategoryCode'.

Case 3: explanatory variables include 'cvvNotSame', 'amountOver', 'posEM_new', 'hour', 'transactionAmount', 'availableMoney', 'cardPresent', 'transactionType'.

## Model 1 - Logistic regression (Case 2)
Implemented train-test split, cross validation, model fitting, roc curve visualization.

<img src="https://user-images.githubusercontent.com/64850893/103326030-82dd1a00-4a1c-11eb-918e-9475ff45ff1a.jpg" width="600" height="300">

## Model 1.2 - Logistic regression after scaling (Case 2)
Used MinMax scaling before applying the model.

<img src="https://user-images.githubusercontent.com/64850893/103326071-ba4bc680-4a1c-11eb-95a1-a29c39670e93.jpg" width="600" height="300">

Compare 2 roc_auc results before and after scaling the dataframe, it can be concluded that scaling the dataframe increased the value of the metric. Therefore, I used the scaling dataframe in the following modeling experiments.

## Model 2 - XGBoost （Case 2）
In addition to the previous prodecures, I utilized the Grid Search method to conduct the hyperparameter tuning (learning rate, number of estimators), selected the optimal combination of parameters, and plotted the roc_auc curve.

<img src="https://user-images.githubusercontent.com/64850893/103326158-08f96080-4a1d-11eb-924a-d5d016c69cdb.jpg" width="600" height="300">

## Model 3 - lightGBM (Case 2)
Similar to the XGBoost modeling, I used the Grid Search approach to conduct the hyperparameter tuning (learning rate, number of estimators, max_depth) for the lightGBM model, chose the optimal combination of parameters, and plotted the roc_auc curve.

<img src="https://user-images.githubusercontent.com/64850893/103326210-40680d00-4a1d-11eb-9754-b1959e0d7491.jpg" width="600" height="300">

## Conclusion 

After comparing the roc_auc of the 3 models in 3 cases, both XGBoost and lightGBM with the feature combination case 2, achieved the highest roc_auc 0.76, increasing the base value by 7%. However, a disadvantage of XGBoost was that its running time was relatively slow. Therefore, lightGBM with case 2 was the optimal model. To further investigate this model, I produced the following feature importance plot. It's straightforward that "transactionAmount", "availableMoney", "posEM_new" were the key factors that would determine whether a transaction was a fraud or not.

<img src="https://user-images.githubusercontent.com/64850893/103326620-3b0bc200-4a1f-11eb-8936-057897973506.jpg" width="700" height="400">


## Future

In the future investigation, I will attempt to experiment more kinds of feature combination. Additionally, more machine learning models with hyperparameter tuning will be applied.
