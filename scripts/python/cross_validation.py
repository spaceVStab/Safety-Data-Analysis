#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:46:52 2017

@author: ajinkya
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
#
#
#### Label encoding state
#state_le = preprocessing.LabelEncoder()
#state_le.fit(train.state)
#train.state=state_le.transform(train.state)
#test.state=state_le.transform(test.state)   
#    
#### Label encoding area code
#area_le = preprocessing.LabelEncoder()
#area_le.fit(train.area_code)
#train.area_code=area_le.transform(train.area_code)
#test.area_code=area_le.transform(test.area_code)   

### Class Weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(train.churn), train.churn)
class_weight_dict=dict(zip(np.unique(train.churn),class_weight))

y_train=train.churn.values
x_train=train.drop("churn",axis=1).values

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

rf=RandomForestClassifier(max_depth= 4, max_features= 'sqrt', min_samples_leaf= 2,n_estimators=200,oob_score=True,n_jobs=-1,random_state=8)
knn=KNeighborsClassifier(n_neighbors=9,n_jobs=-1,weights='distance',p=1)
ada=AdaBoostClassifier(n_estimators=200,learning_rate= 0.1,random_state=8)
gnb=GaussianNB()
xgb= XGBClassifier(colsample_bylevel= 1,
                      n_estimators=200,
                     colsample_bytree= 0.8,
                     learning_rate= 0.1,
                     max_depth= 6 ,seed=8,reg_alpha= 1, reg_lambda= 0.5)
lgr=LogisticRegression(C= 10,random_state=8)
svm=SVC(C= 1000, gamma= 0.0001,random_state=8)

from sklearn.model_selection import cross_validate

scoring = {'acc': 'accuracy',
           'f1': 'f1',
           'precision': 'precision',
           'recall': 'recall',
           'roc_auc':'roc_auc'
           }

clfs=[rf,knn,ada,xgb,gnb,lgr,svm]
names=['Random Forest','K Nearest Neighbors','AdaBoost','XGB','Gaussian Naive Bayes','LogisticRegression','Support Vector Machine']
cv=5

average_scr={}
fold_score={}
for clf, name in zip(clfs, names):
    print (name)
    scores = cross_validate(clf, x_train, y_train, scoring=scoring,
                         cv=cv, return_train_score=True,n_jobs=-1)
    avg_score={}
    if 'score_time' in scores: del scores['score_time']
    for metric_name in scores.keys():
        avg_score[metric_name] = np.average(scores[metric_name])
#    scores['clf']=[name]*cv
    average_scr[name]=avg_score
    fold_score[name]=scores
    
avg_scr=pd.DataFrame.from_dict(average_scr, orient='index')
#f_scr=pd.DataFrame.from_dict(fold_score, orient='index')
f_scr_temp=pd.DataFrame(fold_score)

f_scr=f_scr_temp['AdaBoost'].apply(pd.Series).stack().reset_index(level=-1, drop=True).astype(float).to_frame()
f_scr.rename(columns ={0: 'AdaBoost'}, inplace =True)

for col in [x for x in list(f_scr_temp) if x != "AdaBoost"]:
    temp=f_scr_temp[col].apply(pd.Series).stack().reset_index(level=-1, drop=True).astype(float).to_frame()
    temp.rename(columns ={0: col}, inplace =True)
    f_scr = pd.concat([f_scr, temp], axis=1, join_axes=[f_scr.index])
    


f_scr.reset_index(inplace=True)
f_scr.rename(columns={'index': 'Metrics'},inplace=True)



#import matplotlib.pyplot as plt
#plt.figure();
#
#bp = f_scr.boxplot(by='Metrics',figsize=(10,10))
#
#f_scr.plot(kind='box',by='Metrics',subplots=True,title='Algorithm Comparison',legend=True)


f_scr.to_csv("CV_FOLD_METRICS_after_tuned_cleaned.csv")
avg_scr.to_csv("AVG_CV_FOLD_METRICS_after_tuned_cleaned.csv")

