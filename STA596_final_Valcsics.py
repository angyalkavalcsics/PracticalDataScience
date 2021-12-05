# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 21:38:36 2021

@author: angya
"""
###############################################################################
# Packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 
from sklearn import ensemble, preprocessing
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import accuracy_score as acc
###############################################################################
'''
You should submit a one page summary (including any tables, figures, formulas, 
etc.) in report format (no computer code) to present your analysis and results. 
In addition, you should submit your python code for me to execute.

At least some descriptive analysis for each data set.
At least one relevant plot for each data set.
Perform at least one hypothesis test.
'''
###############################################################################
# Urban Land Cover data
###############################################################################
# Read in data
path = "C:/Users/angya/.spyder-py3/STA596_final_data/Land Cover"
land_train = pd.read_csv(path+"/training.csv")
land_test = pd.read_csv(path+"/testing.csv")
'''
Initial thoughts: perform classification + feature selection.
This could be a classification random forest with tuned hyperparameters. 
I would argue that rf falls in the feature selection category as well.

This or the next data may be the best choice for resampling/hypothesis test.
Thinking back to the iris problem--check variance. 
'''
# find unique classes
np.unique(land_train['class'])
# map classes to numeric representation in both train/test data
le = preprocessing.LabelEncoder()
le.fit(land_train['class'])
land_train['class']=le.transform(land_train['class'])
le.fit(land_test['class'])
land_test['class']=le.transform(land_test['class'])
# split into response/data matrix
X_train = land_train.iloc[:,1:]
y_train = land_train.iloc[:, 0]
X_test= land_test.iloc[:,1:]
y_test = land_test.iloc[:, 0]
###############################################################################
# Random Forest classification 
# Tune hyperparameters of RF using GridSearchCV
param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[10, 20, 30, 40, 50],
    "max_features":[0.6,0.8,0.10,0.12,0.14,0.16, 0.18, 0.20, 0.25, 0.30]
}

clf = ensemble.RandomForestClassifier()

clf_tuned = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=5,
                            n_jobs=-1,
                            verbose=2)

clf_tuned.fit(X_train, y_train)
clf_best = clf_tuned.best_estimator_
clf_best
'''
RandomForestClassifier(max_depth=30, max_features=0.6, n_estimators=200)
'''
y_train_pred = clf_best.predict(X_train)
y_test_pred = clf_best.predict(X_test)

# Note: due to randomness, these may vary when you run it. 
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred)) 
# 100%
print('Training accuracy on selected features: %.3f' % acc(y_test, y_test_pred)) 
# 80.1%
# Perhaps this overfit the training set.

# Find/sort most important features
feature_importances = np.mean([
    tree.feature_importances_ for tree in clf_best.estimators_
], axis=0)

feat_importance = []
feature_names = []
for i in np.argsort(feature_importances)[::-1]:
    feature_names.append(X_train.columns[i])
    feat_importance.append(feature_importances[i])
    print(f'\t{X_train.columns[i]}: {feature_importances[i]:.3f}')
  
''' just some: (to validate my next move)
    NDVI: 0.120
	NDVI_40: 0.078
	Bright_80: 0.046
	NDVI_60: 0.042
	Bright_100: 0.025
	Mean_G: 0.023
	NDVI_80: 0.022
	SD_G: 0.021
	BrdIndx_80: 0.021
	Area_60: 0.020
	Mean_R_80: 0.019
	ShpIndx_140: 0.017
	Mean_NIR_40: 0.017
	...
'''
###############################################################################  
# Let's use the tuned model above in step forward feature selection
'''
Why? Well, above I found the important features but correlated features 
will be given equal or similar importance. My goal with using stepwise 
forward selection in conjunction with this tuned RF model is to cut out
some of these correlated features. My idea is that a variable which is 
correlated with an existing variable in the model will not be selected
since it isn't adding anything new to the prediction power. 
'''
# Build step forward feature selection
clf_sfs = ensemble.RandomForestClassifier(max_depth=30, max_features=0.6, n_estimators=200)
sfs1 = sfs(clf_sfs,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)
# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)
# Which features did it pick?
feat_cols = list(sfs1.k_feature_idx_)
print(X_train.columns[feat_cols])
'''
Index(['Rect', 'BordLngth', 'NDVI_40', 'ShpIndx_100', 'Mean_NIR_140'], 
      dtype='object')
'''
# Reduce features to the 5 best found using SFFS
clf_sfs.fit(X_train.iloc[:, feat_cols], y_train)

y_train_pred = clf_sfs.predict(X_train.iloc[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))
# 100%
y_test_pred = clf_sfs.predict(X_test.iloc[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))
# 75.5%
'''
Compared to the RF trained on the full set of variables, this does have a 
reduced test accuracy of approximately 4.6 percent. Depending on our goal,
this may or may not be beneficial to us. If training a NN with all of the 
variables is costly, then just taking these 5 important variables would work. 

Unfortunately the SFS algorithm does take a long time to run, even with only
5-fold cross validation. Ideally, I would find the optimal k_features according
to test accuracy but it is taking too long.

Moreover, I did spend some time increasing k_features to 8 but this actually
decreased test accuracy. 
'''
# Find/sort most important features
feature_importances = np.mean([
    tree.feature_importances_ for tree in clf_sfs.estimators_
], axis=0)

reduced_feat_importance = []
feature_names = []
for i in np.argsort(feature_importances)[::-1]:
    feature_names.append(X_train.iloc[:, feat_cols].columns[i])
    reduced_feat_importance.append(feature_importances[i])
    print(f'\t{X_train.iloc[:, feat_cols].columns[i]}: \
          {feature_importances[i]:.3f}')
    
'''
	NDVI_40: 0.366
	Mean_NIR_120: 0.272
	BordLngth: 0.154
	ShpIndx_100: 0.134
	Rect: 0.074
'''
# Feature Importance plot
fig, ax = pyplot.subplots(figsize=(12, 7.5))
pyplot.rcParams['font.size'] = '12'
ax.barh(feature_names, reduced_feat_importance)
pyplot.xlabel('Importance')
pyplot.ylabel('Feature')
ax.set_title('Random Forest Feature Importance', fontsize=16)
pyplot.show()

'''
What has been learned at this point? 

Of the 5 most important variables according to RF + sequential feature 
selection, 3 are shape variables and 2 are spectral variables. 

My limited knowledge required that I research these spectral variables. 
The most important feature, NDVI_40 assesses whether or not the target being 
observed contains live green vegetation. Recall that two of the class 
variables are tree and grass. I would assume that the variable Rect 
would do well with buildings and cars--with the help of border length and 
shape index. Even after research I'm not sure I understand what sort of info
a near infrared sensor captures. It seems to measure (mean?) distance to target 
surfaces? Perhaps this can help classify the flat land cover classes like
pool and concrete. Of course all these variables are working together,
I am only trying to make sense of the variables for my own understanding.
'''
##################################################################
# Pollution and Mortality Rate data
###############################################################################
# Read in data
path = "C:/Users/angya/.spyder-py3/STA596_final_data/Pollution"
# this way ensures that we get all 16 column values per row
pollution = open(path+"/Pollution.txt", 'r')
Lines = pollution.readlines()
build = []
for i in range(len(Lines) - 1):
    first = Lines[i].split()
    second = Lines[i+1].split()
    temp = first + second
    build.append(temp)
names = ['PREC', 'JANT', 'JULT', 'OVR65', 'POPN', 'EDUC', 'HOUS', 'DENS',
         'NONW', 'WWDRK', 'POOR', 'HC', 'NOX', 'SO', 'HUMID', 'MORT']
pollution_df = pd.DataFrame(build, columns = names)
'''
Initial thoughts: there are definitely some non-linear terms that we could
fit a spline to here. Prediction of mortality rate would be valuable. 
Possibly linear regression with Ridge pentaly and CV, see if there are 
non-linear relationships using pairplot, and go from there.
'''

###############################################################################
# Zoology data
###############################################################################
# Read in data
path = "C:/Users/angya/.spyder-py3/STA596_final_data/Zoo"
zoo = pd.read_csv(path+"/zoo.csv")
'''
Initial thoughts: use a Neural Network for this. Perhaps some type of network 
analysis or clustering. 
'''

###############################################################################
