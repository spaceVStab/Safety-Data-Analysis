#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 19:01:56 2017

@author: ajinkya
"""

import pandas as pd
import numpy as np
from sklearn.utils import class_weight
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing

train=pd.read_csv("Data/churnTrainNew.csv")
test=pd.read_csv("Data/churnTestNew.csv")

#train.drop(["Id","phone_number"],axis=1,inplace=True)
#test.drop(["Id","phone_number"],axis=1,inplace=True)
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
testX=test.values

#from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

rf=RandomForestClassifier(n_estimators=50,oob_score=True,n_jobs=-1,random_state=8)
#cat=CatBoostClassifier()
ada=AdaBoostClassifier(random_state=8)
xgb= XGBClassifier(seed=8)

rf_feature = rf.fit(x_train,y_train).feature_importances_
rf_feature=list(rf_feature)

#cat_feature = cat.fit(x_train,y_train).feature_importance_
#cat_feature = cat.get_feature_importance(X=x_train,y=y_train)
#cat_feature=list(cat_feature)


        
ada_feature = ada.fit(x_train,y_train).feature_importances_
ada_feature=list(ada_feature)


xgb_feature = xgb.fit(x_train,y_train).feature_importances_
xgb_feature=list(xgb_feature)


cols = train.drop("churn",axis=1).columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
#     'CatBoost  feature importances': cat_feature,
     'AdaBoost feature importances': ada_feature,
     'XGB feature importances': xgb_feature
    })


feature_dataframe = feature_dataframe.set_index('features')

for col in list(feature_dataframe):
    feature_dataframe[[col]].sort_values(by=col,ascending=1).plot(kind='barh')

#feature_dataframe.plot(kind='bar',subplots=True,title='Feature importance',legend=False,layout=(2,2),figsize=(10,10))




