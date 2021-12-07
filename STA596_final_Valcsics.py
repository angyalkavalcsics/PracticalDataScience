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
import seaborn as sb
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, KFold
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn import ensemble, preprocessing
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import accuracy_score as acc
from sklearn import linear_model, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import scipy.spatial.distance as ssd
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
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

This or the next data may be the best choice for resampling/hypothesis test.
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

# Descriptive analysis
land_train.describe()
'''
            class     BrdIndx         Area       Round  ...   Assym_140    NDVI_140  BordLngth_140    GLCM3_140
count  168.000000  168.000000   168.000000  168.000000  ...  168.000000  168.000000     168.000000   168.000000
mean     3.839286    2.008512   565.869048    1.132976  ...    0.615357    0.014583     983.309524  1275.292917
std      2.452740    0.634807   679.852886    0.489150  ...    0.239900    0.153677     880.013745   603.658611
min      0.000000    1.000000    10.000000    0.020000  ...    0.070000   -0.360000      56.000000   336.730000
25%      2.000000    1.537500   178.000000    0.787500  ...    0.460000   -0.080000     320.000000   817.405000
50%      4.000000    1.920000   315.000000    1.085000  ...    0.620000   -0.040000     776.000000  1187.025000
75%      6.000000    2.375000   667.000000    1.410000  ...    0.810000    0.120000    1412.500000  1588.427500
max      8.000000    4.190000  3659.000000    2.890000  ...    1.000000    0.350000    6232.000000  3806.360000

[8 rows x 148 columns]
'''
land_train.head()
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
###############################################################################
# Hypothesis: test for difference of means
X = pd.concat([X_train, X_test])
lab = pd.concat([y_train, y_test])

# get  unique  labels
unilab = np.unique(lab)
# number  of  unique  labs
n_labs = len(unilab)
X.columns
y = X['NDVI_40']
# check  for  equal  variances
var_gr = np.zeros(n_labs)
for j in  range(n_labs):
    var_gr[j] = np.var(y[lab == unilab[j]])
    
# no  group  variance is more  than  
# twice  any  other?
np.max(var_gr) > 2*np.min(var_gr) 
# True so to test for difference of means we need to bootstrap

def F(y, lab):
    N = len(y)
    [uni, nj] = np.unique(lab, return_counts = True)
    K = len(nj)
    ybar = np.mean(y)
    ybar_gr = np.zeros(K)
    for k in range(K):
        ybar_gr[k] = np.mean(y[lab == uni[k]])
    ssmod = np.sum(nj*np.power(ybar_gr - ybar, 2))
    sserr = 0
    for k in range(K):
        sserr = sserr + np.sum(np.power(y[lab == uni[k]] - ybar_gr[k], 2))
    return((ssmod/(K-1))/(sserr/(N-K)))
Fstat = F(y, lab)
# Bootstrapping  the  hypothesis  test
B = 10000
F_boot = np.zeros(B)
for j in  range(B):
    lab_boot = np.random.permutation(lab)
    F_boot[j] = F(y, lab_boot)
    
p_boot = np.mean(F_boot > Fstat)
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
Initial thoughts: may be able to find some non-linear terms here. 
Prediction of mortality rate would be valuable. 
Possibly linear regression with Ridge pentaly, see if there are 
non-linear relationships using polynomial regression, and go from there.
'''
# Descriptive analysis
pollution_df.describe()
'''
       PREC JANT JULT OVR65 POPN  EDUC  ...  POOR    HC  NOX   SO HUMID  MORT
count   119  119  119   119  119   119  ...   119   119  119  119   119   119
unique   75   52   48    83   52    85  ...    81    60   83  103    71   111
top     36.  30.  72.    1.  56.  11.4  ...  3.32  11.1   4.   1.   56.  45.5
freq      5    6   11     6   11     6  ...     4     6    7    6    11     2

[4 rows x 16 columns]
'''

# Some of these variables, such as JULT, POPN, and HUMID do not vary so much
# from year to year. However, the response variable does. See that out of 119 
# values, there are 111 unique values. 

pollution_df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 119 entries, 0 to 118
Data columns (total 16 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   PREC    119 non-null    object
 1   JANT    119 non-null    object
 2   JULT    119 non-null    object
 3   OVR65   119 non-null    object
 4   POPN    119 non-null    object
 5   EDUC    119 non-null    object
 6   HOUS    119 non-null    object
 7   DENS    119 non-null    object
 8   NONW    119 non-null    object
 9   WWDRK   119 non-null    object
 10  POOR    119 non-null    object
 11  HC      119 non-null    object
 12  NOX     119 non-null    object
 13  SO      119 non-null    object
 14  HUMID   119 non-null    object
 15  MORT    119 non-null    object
dtypes: object(16)
memory usage: 15.0+ KB
'''

# need to convert these to numeric
pollution_df = pollution_df.apply(pd.to_numeric)

# pair plot
sb.pairplot(pollution_df) # doesn't look very helpful/ no polynomial rel.

# split into train/test -- 20% reserved for testing
X_train, X_test, y_train, y_test = train_test_split(\
        pollution_df.iloc[:, 0:15], pollution_df.iloc[:, 15], test_size=0.2)
###############################################################################
# Ridge Regression and tuning of L2 penalty
model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(1000, 2000, 1) 
# interval endpoints picked by trial/error
# define search
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
search = GridSearchCV(model, grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# ignore the negative sign in the score -- for optimization purposes only
# perform the search
results = search.fit(X_train, y_train)
# summarize
print('MSE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# Alpha is so large, this may not be the best model for the data.
# This is because a large alpha will shrink coefficients near zero.
# We know that as \alpha \to \infty, the coefficients approach 0. 

# Let's get a visual to see if this is the case
best_model = Ridge(alpha=1733)
best_model.fit(X_train, y_train)
best_model.coef_

c = pd.DataFrame(best_model.coef_, index = X_train.columns, columns= ['coefs'])
ridge_feature = c.index[np.nonzero(np.array(c))[0]]
ridge_coefs = c.iloc[np.nonzero(np.array(c))[0]]

fig, ax = pyplot.subplots(figsize=(12, 7.5))
pyplot.rcParams['font.size'] = '12'
ax.barh(list(ridge_feature), list(ridge_coefs.iloc[:,0]))
pyplot.xlabel('Coefficient')
pyplot.ylabel('Feature')
ax.set_title('Ridge Feature Importance', fontsize=16)
pyplot.show()

# Actually, looks okay. Let's see the train/test MSE.
y_train_pred = best_model.predict(X_train)
np.mean((y_train - y_train_pred)**2)
# 1272.56
y_test_pred = best_model.predict(X_test)
np.mean((y_test - y_test_pred)**2)
# 3007.48

'''
What has been learned from these results? 
The average tempurature in Jan, July, and avg annual precip.
have the most influence on mortality rate. Then ext most important
variables are avg. household size, med. school years completed,
non-white pop., % employed white collar, % poor, nitric oxides, and
annual humidity. 

Climate such as heat waves, cold, and heavy rain have all been
proven to cause accidents or result in more deaths. Humidity can
help viruses spread or exacerbate effects of climate. Moreover,
the societal variables shown to be important in this model have
been proven to have an affect on mortality as well. Lastly,
Nitric oxide is colourless and is oxidised in the atmosphere 
to form nitrogen dioxide which is an acidic and highly corrosive 
gas that can affect mortality.
'''
###############################################################################
# can we do better? 
# If we do not care for interpretation and only prediction of 
# mortality--a neural network would surely provide better
# results than the Ridge model above. 
###############################################################################
# First though, I read in a paper while researching the variables above
# that sulfur dioxide has a non-linear relationship with mortality. 
# Clearly, sulfur dioxide is dangerous, why did our model not pick up 
# on this? Can we show a non-linear relationship using our data?
y = np.asarray(pollution_df["MORT"])
x = np.asarray(pollution_df["SO"])

order = np.argsort(x)
x = x[order]
y = y[order]

nf = len(y)
kf = KFold(n_splits=nf,shuffle=True)
ds = np.linspace(2, 6, 5)

cv_err = np.zeros(len(ds))
for d in range(len(ds)):
    for train_index, test_index in kf.split(range(nf)):
        polyreg = make_pipeline(
        PolynomialFeatures(degree=d),
        LinearRegression()
        )

        polyreg.fit(x.reshape(-1,1), y)

        # prediction
        yhat = polyreg.predict(x.reshape(-1,1))
        cv_err[d] = np.mean(np.power(y[test_index]-yhat,2))
opt_deg = ds[np.argmin(cv_err)] # 2

polyreg = make_pipeline(
        PolynomialFeatures(degree=int(opt_deg)),
        LinearRegression()
        )

polyreg.fit(x.reshape(-1,1), y)

# prediction
yhat = polyreg.predict(x.reshape(-1,1))

pyplot.plot(x, y,'k.')
pyplot.plot(x,yhat,'b-')

# I'm not sure about this. The data is very much hot/cold with its measurements
# for sulfur oxide. Perhaps as spline would fit this better.

import scipy.interpolate as interpolate
t, c, k = interpolate.splrep(x, y, s=100, k=3, t =[4, 6, 8, 10])

N = 100
xmin, xmax = x.min(), x.max()
xs = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

pyplot.plot(x, y, 'k.')
pyplot.plot(xs, spline(xs), 'b')

# It definitely looks like a spline with degree 3 would predict this
# (have lower MSE) better than the polynomial regression with degree 2. 
###############################################################################
# Fit a NN to pollution data
m = neural_network.MLPRegressor(
    hidden_layer_sizes=(10),
    learning_rate_init = 0.001,
    max_iter = 10000)

m.fit(X_train,y_train)
# mean test error
1 - m.score(X_test, y_test) # 0.01
# a 1 percent test error is not bad at all
###############################################################################
# Zoology data
###############################################################################
# Read in data
path = "C:/Users/angya/.spyder-py3/STA596_final_data/Zoo"
zoo = pd.read_csv(path+"/zoo.csv")
'''
Initial thoughts: Perhaps some type of network 
analysis or clustering. 
'''
# Descriptive analysis
zoo.describe()
'''
             hair    feathers        eggs  ...    domestic     catsize  class_type
count  144.000000  144.000000  144.000000  ...  144.000000  144.000000  144.000000
mean     0.305556    0.138889    0.708333  ...    0.125000    0.416667    3.430556
std      0.462250    0.347038    0.456116  ...    0.331873    0.494727    2.137421
min      0.000000    0.000000    0.000000  ...    0.000000    0.000000    1.000000
25%      0.000000    0.000000    0.000000  ...    0.000000    0.000000    1.000000
50%      0.000000    0.000000    1.000000  ...    0.000000    0.000000    3.000000
75%      1.000000    0.000000    1.000000  ...    0.000000    1.000000    5.250000
max      1.000000    1.000000    1.000000  ...    1.000000    1.000000    7.000000

[8 rows x 17 columns]
'''
zoo.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 144 entries, 0 to 143
Data columns (total 18 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   animal_name  144 non-null    object
 1   hair         144 non-null    int64 
 2   feathers     144 non-null    int64 
 3   eggs         144 non-null    int64 
 4   milk         144 non-null    int64 
 5   airborne     144 non-null    int64 
 6   aquatic      144 non-null    int64 
 7   predator     144 non-null    int64 
 8   toothed      144 non-null    int64 
 9   backbone     144 non-null    int64 
 10  breathes     144 non-null    int64 
 11  venomous     144 non-null    int64 
 12  fins         144 non-null    int64 
 13  legs         144 non-null    int64 
 14  tail         144 non-null    int64 
 15  domestic     144 non-null    int64 
 16  catsize      144 non-null    int64 
 17  class_type   144 non-null    int64 
dtypes: int64(17), object(1)
memory usage: 20.4+ KB
'''
# How is the class type distributed?
pyplot.hist(zoo['class_type'])

'''
Thinking about similarity measures for this... 

We have just binary vectors--I will one-hot encode the legs variable.

After some research, perhaps the jaccard coefficient would work well 
for this binary data. For each variable/data point, the value is 1 if the animal
has the quality we are interested in and 0 if not--i.e. positive/negative states. 
This means that our binary variables are asymmetric attributes, I would argue
that we only care if the animal has the quality that we are interested in.

Let's see how similar the variables are first. 
'''
# one-hot encode the legs variable 
zoo_reduced = zoo.iloc[:, 1:]
one_hot = pd.get_dummies(zoo_reduced, columns = ['legs'])
one_hot = one_hot.T
# heatmap of jaccard
jac_sim = 1 - pairwise_distances(one_hot, metric = "hamming")
jac_sim = pd.DataFrame(jac_sim, one_hot.columns, columns=one_hot.columns)

fig = pyplot.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
sb.heatmap(jac_sim, annot=True, annot_kws={"size": 8}, 
            xticklabels=jac_sim.columns.values,
            yticklabels=jac_sim.columns.values)
pyplot.title("Jaccard Similarity of Zoo variables")
###############################################################################
# Hierarchical clustering of zoo data
# I will pass in a distance matrix using Jaccard dissimilarity (so diag is 0)

one_hot = one_hot.T
jac_sim = pairwise_distances(one_hot, metric = "hamming")

# first produce a dendrogram
# convert the redundant n*n square matrix form into a condensed nC2 array
d = ssd.squareform(jac_sim)
Z = linkage(d, 'single')

pyplot.figure(figsize=(10, 7))
dendrogram(Z,
            orientation='top',
            labels=None,
            distance_sort='descending',
            show_leaf_counts=True)
pyplot.show()
# This had trouble -- only found 6 classes instead of 7

# now, I want the actual labels
model = AgglomerativeClustering(affinity='precomputed', n_clusters=7, linkage='single').fit(jac_sim)
print(model.labels_)
np.shape(model.labels_)

# use pca to produce a plot
pca = PCA(2) 
pca.fit(one_hot) 
pca.components_  

# i'm counting this as utilizing svd (hopefully that is okay)
U = pca.transform(one_hot)
np.shape(U)

pyplot.scatter(U[:,0],U[:,1],c= model.labels_, cmap='rainbow')

y_act = zoo_reduced['class_type'] - 1
np.mean(np.power(y_act - model.labels_, 2)) # 11.0

# Generate plot
pyplot.hist([y_act, model.labels_], label=['actual classification', 'predicted'])
pyplot.legend(loc='upper right')
pyplot.title("Zoo Classification via Clustering")
pyplot.xlabel("Class Label")
pyplot.ylabel("Frequency")
pyplot.show()

'''
Thoughts: Not too bad. The model certainly struggled to classify
the 4th and 6th classes -- amphibians and mollusks (?)

Instead it overclassified animals into the 3rd class, fish. 
Which, to some extent, is forgivable. There is not much of a difference
in characteristics between these types of animals and there are not many
data points.

On the other classes however the model got close to or perfect classification. 
'''
###############################################################################
# Network analysis of zoo data
'''
Here is my idea: if jaccard similarity between entities (animals) is >=
to the cut off value above, we enter a 1 in the adjacency matrix and 0 otherwise. 

May want to experiment with this cut off a little, i'll start with using the
min of these and maybe also try the median of the medians.
'''

cutoff = np.min(pd.DataFrame(jac_sim).median())
cutoff

cutoff = np.mean(pd.DataFrame(jac_sim).median())
cutoff

import statistics
cutoff = statistics.median(pd.DataFrame(jac_sim).median())
cutoff

cutoff = np.mean(pd.DataFrame(jac_sim).mean())
cutoff

cutoff = 0.5
cutoff = 0.45

# The cutoff controls the density of the model, it is 
# kind of like a hyperparameter that needs to be optimized/chosen 
# carefully. I'm not sure I like any of these but the last one.
# A cut off of 0.5 is too high -- the graph becomes disconnected.
# Perhaps, somewhere between 0.4 and 0.5 would be best. 

A = (jac_sim >= cutoff)*1
G = nx.convert_matrix.from_numpy_matrix(A)
nx.draw(G,nx.spring_layout(G), node_size = 20)

# pip install python-louvain

import community
# find the clusters 
partition = community.best_partition(G)
s = set(partition.values())
num_clusters_found = len(s) 
num_clusters_found
# we need a vector with the cluster ids to show clusters using color in a plot
color_vec = list(partition.values())
# to plot labels, you need a dictionary
labels2 = {a:a for a in list(nx.nodes(G))} 
nx.draw(G,pos=nx.spring_layout(G),node_color=color_vec,labels=labels2,font_size=10,alpha=.6 )

# color_vec is our predicted class for each node
np.mean(np.power(y_act - color_vec, 2)) # 6.368
(y_act - color_vec) == 0
# When I ran the algorithm, I got 5 clusters. 
# It is surprising to me that this mse is much lower
# and I wonder about the validity of what I'm doing

# Generate plot
pyplot.hist([y_act, color_vec], label=['actual classification', 'predicted'])
pyplot.legend(loc='upper right')
pyplot.title("Zoo Classification via Network Clustering")
pyplot.xlabel("Class Label")
pyplot.ylabel("Frequency")
pyplot.show()

# Specifically, this plot worries me
pyplot.scatter(U[:,0],U[:,1], c = color_vec, cmap='rainbow')

# plot degree distribution
def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    pyplot.hist(degrees)
    pyplot.show()

plot_degree_dist(G)

num_triangles = int(sum(nx.triangles(G).values()) / 3)
num_triangles # 21842
# Hence there are 21842 triangles in the data. This means that 
# This means that there are significant amount of triad relationships
# in the data. 
