# Credit_Risk_Analysis
To Build and evaluate several machine learning models or algorithms to predict credit risk using Python and Scikit Libraries.

## Overview of the Analysis
Loans are the essential part of the modern system.On one hand loan creates revenues and other hand there is a risk that borrower wont repay the loans and bank will lose the money.Banks have traditionally relied on the measures like credit scores,income and collateral assests to assess the lending risk.The rise of financial technology enabled lenders to use the Machine Learning to analyze the risk.Machine Learning can process large amount of data to arrive to a single decision whether or not the lenders to approve the loan application.We will use python and scikit learn libraries to build and evaluate several machine learning models to predict credit risk.we will compare the strengths and weakness of different machine learning models.We will assess how well a model classifies and predicts data.we will be using skills like data munching and resampling.

## Purpose of the Analysis
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. We will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we will use a combinatorial approach of over and undersampling using the SMOTEENN algorithm. Next, we will compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once they are done,we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Resources Used
*DataSources*:  [LoanStats_2019Q1.csv](https://github.com/fathi129/Credit_Risk_Analysis/blob/master/LoanStats_2019Q1.csv)<br>
*Software used*: Jupyter Notebook <br>
*Language*: Python<br>
*Libraries*: Scikit-learn,imbalanced-learn.<br>

## Results
### Deliverable 1: Use Resampling Models to Predict Credit Risk
Using imbalanced-learn and scikit-learn libraries, we will evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, we will use the oversampling RandomOverSampler and SMOTE algorithms, and then we will use the undersampling ClusterCentroids algorithm. Using these algorithms, we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.First we will create the training variables by converting the string values into numerical ones using the get_dummies() method.Create the target variables.Check the balance of the target variables.Next we will begin resampling the training data.

## OverSampling 
## RandomOverSampler algorithm 
Random oversampling involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset.After resampling the training data we get the following results
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/balance_ros.png"  width = 900><br>
Confusion Matrix:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/cm_ros.png"  width = 900><br>
Classification Report:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/report_ros.png"  width = 900><br>
- Balance Accuracy Score: 64.9%
- High-Risk Precision: 0.01
- Low-Risk Precision: 1.00
- High-Risk Recall: 0.62
- Low-Risk Recall: 0.68
- High-Risk F1 Score: 0.02
- Low-Risk F1 Score: 0.81

## SMOTE algorithm
SMOTE works by selecting examples that are close in the feature space, drawing a line between the examples in the feature space and drawing a new sample at a point along that line.The approach is effective because new synthetic examples from the minority class are created that are plausible, that is, are relatively close in feature space to existing examples from the minority class.A general downside of the approach is that synthetic examples are created without considering the majority class, possibly resulting in ambiguous examples if there is a strong overlap for the classes.After resampling the training data we get the following results<br>
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/balance_smote.png"  width = 900><br>
Confusion Matrix:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/cm_smote.png"  width = 900><br>
Classification Report:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/report_smote.png"  width = 900><br>
- Balance Accuracy Score:64.43%
- High-Risk Precision: 0.01
- Low-Risk Precision: 1.00
- High-Risk Recall: 0.63
- Low-Risk Recall: 0.66
- High-Risk F1 Score: 0.02
- Low-Risk F1 Score: 0.79



## UnderSampling
## ClusterCentroids algorithm
Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.After resampling the training data we get the following results:<br>
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/balance_cc.png"  width = 900><br>
Confusion Matrix:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/cm_cc.png"  width = 900><br>
Classification Report:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%201/report_cc.png"  width = 900><br>
- Balance Accuracy Score:52.93%
- High-Risk Precision: 0.01
- Low-Risk Precision: 1.00
- High-Risk Recall: 0.61
- Low-Risk Recall: 0.45
- High-Risk F1 Score: 0.01
- Low-Risk F1 Score: 0.62


## Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk
A combinatorial approach of over and undersampling algorithm is used.we will determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, we res will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
## SMOTEENN ALGORITHM
SMOTEENN Algorithm,this method combines the SMOTE ability to generate synthetic examples for minority class and ENN ability to delete some observations from both classes that are identified as having different class between the observation’s class and its K-nearest neighbor majority class.After resampling the training data we get the following results:<br>
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/balance_smoteen.png"  width = 900><br>
Confusion Matrix:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%202/cm_smoteen.png"  width = 900><br>
Classification Report:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%202/report_smoteen.png"  width = 900><br>
- Balance Accuracy Score:63.7%
- High-Risk Precision: 0.01
- Low-Risk Precision: 1.00
- High-Risk Recall: 0.71
- Low-Risk Recall: 0.56
- High-Risk F1 Score: 0.02
- Low-Risk F1 Score: 0.72
- 
## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
Using your knowledge of the imblearn.ensemble library, you’ll train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
## BalancedRandomForestClassifier
A balanced random forest randomly under-samples each boostrap sample to balance it.After resampling the training data we get the following results:<br>
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/balan%20e_brfc.png"  width = 900><br>
Confusion Matrix:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/cm_brfc.png"  width = 900><br>
Classification Report:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/report_brfc.png"  width = 900><br>
- Balance Accuracy Score:78.77%
- High-Risk Precision: 0.04
- Low-Risk Precision: 1.00
- High-Risk Recall: 0.67
- Low-Risk Recall: 0.91
- High-Risk F1 Score: 0.07
- Low-Risk F1 Score: 0.95



## EasyEnsembleClassifier
This algorithm is known as EasyEnsemble.The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.After resampling the training data we get the following results:<br>
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/balance_eec.png"  width = 900><br>
Confusion Matrix:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/cm_eec.png"  width = 900><br>
Classification Report:
<img src = "https://github.com/fathi129/Credit_Risk_Analysis/blob/master/Screenshots%20of%20Credit%20Risk%20Analysis/Deliverable%203/report_eec.png"  width = 900><br>
- Balance Accuracy Score:92.5%
- High-Risk Precision: 0.07
- Low-Risk Precision: 1.00
- High-Risk Recall: 0.91
- Low-Risk Recall: 0.94
- High-Risk F1 Score: 0.14
- Low-Risk F1 Score: 0.97




