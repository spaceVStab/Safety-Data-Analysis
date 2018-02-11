# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 23:06:34 2017

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
from xgboost import XGBClassifier

#parameters = {'learning_rate':[0.05,0.1,0.5,1], 'colsample_bytree':[0.6,0.8,1],'colsample_bylevel':[0.6,0.8,1]  
#                    ,'max_depth':[4,6,8] }

parameters = {'reg_alpha':[0.1,0.5,1],'reg_lambda':[0.1,0.5,1] }

#xgb=XGBClassifier(n_estimators=100,nthread=4,seed=8)
xgb= XGBClassifier(colsample_bylevel= 1,
                      n_estimators=200,
                     colsample_bytree= 0.8,
                     learning_rate= 0.1,
                     max_depth= 6 ,seed=8)

scoring = {'acc': 'accuracy',
           'f1': 'f1',
           'precision': 'precision',
           'recall': 'recall',
           'roc_auc':'roc_auc'
           }


clf = GridSearchCV(xgb, parameters,verbose=1,scoring=scoring,refit="roc_auc",n_jobs=-1,cv=5)

clf.fit(x_train,y_train)


print(clf.best_score_)
clf.best_params_
