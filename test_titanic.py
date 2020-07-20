#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:42:39 2020

@author: cpprhtn
"""


연습

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("sample_submission.csv")

train.info()
train.isna().sum()
test.isna().sum()


train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'), inplace=True)

train.isna().sum()
train.Age.mean()

test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'), inplace=True)
test.Age.mean()

sns.lmplot('Pclass', 'Age', data = train, fit_reg=True)
train.Pclass.unique()

train['Embarked'].value_counts()

train['Embarked'].fillna('S', inplace=True)

train['Embarked'].value_counts()

train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')
train = pd.get_dummies(train, columns=['Sex'], prefix='Sex')
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')
test = pd.get_dummies(test, columns=['Sex'], prefix='Sex')

train.info()
features = ['Pclass','Sex_female','Sex_male','Age','SibSp','Embarked_C','Embarked_Q','Embarked_S']
target = ['Survived']

X_train, X_test, y_train = train[features], test[features], train[target]

X_train.info()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor


model_dict = {
    'logistic':LogisticRegression(),
    'Decision':DecisionTreeClassifier(),
    'liner': LinearRegression(),
    'rf':RandomForestRegressor(),
    'lgb':lgb.LGBMRegressor()
}


model_result = {}
for key in model_dict.keys():
    model_dict[key].fit(X_train, y_train)
    
    model_result[key] = model_dict[key].predict(X_test)

log_submit=log_submit.drop('Servived',axis=1)
log_submit = submission.copy()
log_submit["Survived"]=model_result['logistic']

Dec_submit = submission.copy()
Dec_submit["Survived"]=model_result['Decision']

del model_result['logistic']
del model_result['Decision']


log_submit.to_csv('log_submit.csv', index =False)
