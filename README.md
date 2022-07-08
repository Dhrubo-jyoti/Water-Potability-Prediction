# Water-Potability-Prediction
This is a Classification model used to predict the potability of water based on its 'Ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity'. This is a self-made project whose inspiration was taken from Kaggle.

## Data Collection 
The data source for this project was collected from kaggle.
Source of the data is: https://www.kaggle.com/datasets/adityakadiwal/water-potability

## EDA
The data set had 3276 columns along with 10 columns. Among the 10 columns, 9 were independent features and 1 was the dependent feature. There was imbalance in the dataset having 1998 rows with data of impotable water samples and 1278 rows with data of potable water samples.
Upon further analyzing the data there were many insights we came across. A key highlight of them are:
1) There are 9 numerical features(Independent Variables), 1 target Variable and NO Categorical features.
2) The dataset had missing values.
3) Many feaures didn't had a normal disribution and has some kind of skewness in it, refering that the data set had OULIERS.
4) The data in the given data set is not scaled in the same order.
5) There was no correlation among he feaures themselves nor with the target variable.
All these insights were gained by plotting graphs such as Boxplot, Histogram, Pairplot, Piechart, etc. and statistically analyzing the features.

## FEATURE ENGINEERING
In this section we dealt with the outliers and missing values. To deal with the outliers we used the Z-SCORE method using which we dropped 116 values and to deal with the missing values we opted to impute the 781, 491 and 162 values from 'Sulfate', 'Ph', 'Trihalomethanes' respectively using KNN IMPUTER.  

## FEATURE SELECTION
To select among all the given features which features to use to build our model we first tried to find the correlation among the features using 'Pearson' method but were not able to find any good correlation among the features themselves or with the target variable. Then we plotted a pairplot to see the relation between the variables and found that there was no clear relation between them and they were distributed very randomly overlapping with each other. For this we tried to find the correlation using 'Spearman' method but there was no correlation between them at all.
From the Pairplot we can also conclude that models like Linear Regression , Logistic Regression will not work. So we decided to use models like KNN, Random Forest , Decision tree, etc. i.e. models which are tree based, distance based or are boosting techniques.

## FEATURE SCALING
All the features are on different scales and to have better accuracy scores on models like KNN, Decision Tree we have to scale them down on same level. We will be using Min-Max Scaler method as we have found out using cross-validation score that the best performing model is KNN.

## BALANCING THE DATASET
The distribution of the 0-class and 1-class was 61% and 39% respectively. We are using SMOTE Technique to synthetically upsample the minority class to get an equal distribution of 0 & 1.

## MODEL BUILDING
Some of the classical distance based, tree based and boosting technique are used to make the predictions. Among them the best performing models based on their Accuracy score without any parameter tuning were RandomForest , XGBoost & KNN with scores of 70.8% , 67.12% & 66.12% respectively.
Then we tried to find their cross-validation-score where we found out that KNN had the best avg. score of 73.6%.

## HYPERPARAMETER TUNING
Upon figuring out that KNN was our best model we used GRIDSEARCH CV to find the best possible combination of parameters to improve the Accuracy score and Recall value of the model. After tuning the model we got an improved Accuracy score value of 73.75% up from the initial value of 66.12% and an improved Recall value of 80.41%.
The final F1-Score of the model was 74.11% and Precision of 68.72% for the 1-class(i.e. the true class)
