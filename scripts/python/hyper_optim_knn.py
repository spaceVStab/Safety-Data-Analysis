# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 23:59:32 2017

@author: batman
"""

import pandas as pd
import numpy as np
from sklearn.utils import class_weight
#from sklearn.model_selection import train_test_split

train=pd.read_csv("Data/churnTrainNew.csv")
test=pd.read_csv("Data/churnTestNew.csv")

ignore=['total_eve_calls', 'total_day_calls',
            'number_vmail_messages']
train.drop(ignore,axis=1,inplace=True)
test.drop(ignore,axis=1,inplace=True)

class_weight = class_weight.compute_class_weight('balanced', np.unique(train.churn), train.churn)
class_weight_dict=dict(zip(np.unique(train.churn),class_weight))

y_train=train.churn.values
x_train=train.drop("churn",axis=1).values
testX=test.values

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    scores = cross_val_score(knn, x_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print ("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, cv_scores)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.show()

##### finding other optimum parameters:

from sklearn.model_selection import GridSearchCV

parameters = {'weights':['uniform','distance'],'p':[1,2]}
knn=KNeighborsClassifier(n_neighbors=5,n_jobs=-1)

scoring = {'acc': 'accuracy',
           'f1': 'f1',
           'precision': 'precision',
           'recall': 'recall',
           'roc_auc':'roc_auc'
           }


clf = GridSearchCV(knn, parameters,verbose=1,scoring=scoring,refit="roc_auc",n_jobs=-1,cv=5)

clf.fit(x_train,y_train)


print(clf.best_score_)
clf.best_params_
