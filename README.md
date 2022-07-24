# Credit_Risk_Analysis
To Build and evaluate several machine learning models or algorithms to predict credit risk using Python and Scikit Libraries.

## Overview of the Analysis
Loans are the essential part of the modern system.On one hand loan creates revenues and other hand there is a risk that borrower wont repay the loans and bank will lose the money.Banks have traditionally relied on the measures like credit scores,income and collateral assests to assess the lending risk.The rise of financial technology enabled lenders to use the Machine Learning to analyze the risk.Machine Learning can process large amount of data to arrive to a single decision whether or not the lenders to approve the loan application.We will use python and scikit learn libraries to build and evaluate several machine learning models to predict credit risk.we will compare the strengths and weakness of different machine learning models.We will assess how well a model classifies and predicts data.we will be using skills like data munching and resampling.

## Purpose of the Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. We will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once they are done,we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Resources Used
*DataSources*: [LoanStats_2019Q1.csv](https://github.com/fathi129/Credit_Risk_Analysis/blob/master/LoanStats_2019Q1.csv)<br>
*Software used*: Jupyter Notebook <br>
*Language*: Python<br>
*Libraries*:Scikit-learn,imbalanced-learn.<br>
