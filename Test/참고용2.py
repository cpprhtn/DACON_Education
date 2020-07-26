#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:23:04 2020

@author: cpprhtn
"""


features = ['Age','Sex', ~~]
target = ['Servied']


X_train, X_test, y_train = train[features], test[features], train[target]

#타이타닉
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

#버스운용
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

model_dict = {
    'liner': LinearRegression(),
    'rf':RandomForestRegressor(),
    'lgb':lgb.LGBMRegressor()
}

model_dict.keys()



model_result = {}
for key in model_dict.keys():
    model_dict[key].fit(X_train, y_train)
    
    model_result[key] = model_dict[key].predict(X_test)
    
    
lr_submit = submission.copy()

lr_submit["Servied"]=model_result['linear']