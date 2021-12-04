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
This could be a classification random forest with tuned hyperparameters and
pca plugged into logistic regression. I would argue that rf falls in the 
feature selection category as well.

This or the next data may be the best choice for resampling/hypothesis test.
Thinking back to the iris problem--check variance. 
'''

###############################################################################
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