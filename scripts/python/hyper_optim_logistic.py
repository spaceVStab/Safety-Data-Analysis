# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 00:20:39 2017

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

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

parameters = {'C':[0.01,0.1,1,10]}
lgr=LogisticRegression(random_state=8)

scoring = {'acc': 'accuracy',
           'f1': 'f1',
           'precision': 'precision',
           'recall': 'recall',
           'roc_auc':'roc_auc'
           }


clf = GridSearchCV(lgr, parameters,verbose=1,scoring=scoring,refit="roc_auc",n_jobs=-1,cv=5)
#clf = GridSearchCV(lgr, parameters,verbose=1,scoring='accuracy',n_jobs=-1,cv=5)

clf.fit(x_train,y_train)


print(clf.best_score_)
clf.best_params_
