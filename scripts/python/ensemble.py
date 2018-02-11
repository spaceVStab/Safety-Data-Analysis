# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:26:54 2017

@author: batman
"""

import pandas as pd
import numpy as np


train_sub=pd.read_csv("Data/churnTrainNew.csv")
test_sub=pd.read_csv("Data/churnTestNew.csv")
ignore=['total_eve_calls', 'total_day_calls',
            'number_vmail_messages']
train_sub.drop(ignore,axis=1,inplace=True)
test_sub.drop(ignore,axis=1,inplace=True)

#train.drop(["Id","phone_number"],axis=1,inplace=True)
#test.drop(["Id","phone_number","churn"],axis=1,inplace=True)
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


y_train=train_sub.churn.values
x_train=train_sub.drop("churn",axis=1).values
y_test_sub=test_sub.churn.values
testX=test_sub.drop("churn",axis=1).values


from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

np.random.seed(0)  # seed to shuffle the train set

n_folds = 5
verbose = True
shuffle = False

X, y, X_submission = x_train, y_train, testX # Or use sklearn train test split to get numpy arrays


skf = list(StratifiedKFold(y, n_folds))

clfs = [
RandomForestClassifier(max_depth= 4, max_features= 'sqrt', min_samples_leaf= 2,n_estimators=200,oob_score=True,n_jobs=-1,random_state=8),
AdaBoostClassifier(n_estimators=200,learning_rate= 0.1,random_state=8),
GaussianNB(),
XGBClassifier(colsample_bylevel= 1,
                      n_estimators=200,
                     colsample_bytree= 0.8,
                     learning_rate= 0.1,
                     max_depth= 6 ,seed=8,reg_alpha= 1, reg_lambda= 0.5),
SVC(C= 1000, gamma= 0.0001,random_state=8,probability=True)
     ]

print ("Creating train and test sets for blending.")

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

### Blending ensemble

for j, clf in enumerate(clfs): 
    print (j, clf)
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print ("Fold", i)
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
    clf.fit(X, y)
    dataset_blend_test[:, j] = clf.predict_proba(X_submission)[:, 1]

pd.DataFrame(dataset_blend_train).to_csv("blended_train.csv")
pd.DataFrame(dataset_blend_test).to_csv("blended_test.csv")

print ("Blending..........")

clf = XGBClassifier(n_estimators=200, seed=8,nthread =4)
clf.fit(dataset_blend_train, y)
y_submission = clf.predict(dataset_blend_test)

### Scoring
from sklearn.model_selection import cross_validate

scoring = {'acc': 'accuracy',
           'f1': 'f1',
           'precision': 'precision',
           'recall': 'recall',
           'roc_auc':'roc_auc'
           }

cv=5

scores = cross_validate(clf, dataset_blend_train, y, scoring=scoring,
                         cv=cv, return_train_score=True,n_jobs=-1)

avg_score={}
if 'score_time' in scores: del scores['score_time']
for metric_name in scores.keys():
    avg_score[metric_name] = np.average(scores[metric_name])


avg_scr=pd.DataFrame.from_dict(avg_score, orient='index')
avg_scr.to_csv("cross_validation_final_result.csv")


from sklearn.metrics import classification_report
report = classification_report(y_test_sub, y_submission)
print(report)
       ########### 
#test_scores = cross_validate(clf, dataset_blend_test, y_test, scoring=scoring,
#                         cv=cv, return_train_score=True,n_jobs=-1)
#
#avg_test_score={}
#if 'score_time' in test_scores: del test_scores['score_time']
#for metric_name in test_scores.keys():
#    avg_test_score[metric_name] = np.average(test_scores[metric_name])
#
#
#avg_scr=pd.DataFrame.from_dict(avg_test_score, orient='index')
#avg_scr.to_csv("test_final_result.csv")

