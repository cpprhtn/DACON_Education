#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:59:13 2020

@author: cpprhtn
"""


import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("sample_submission.csv")



#전처리
train.isna().sum()
test.isna().sum()

train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'), inplace=True)
test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'), inplace=True)
train['Age'].mean()
gender = { "male":0, "female":1}
train['Sex']=train['Sex'].replace(gender)
test['Sex']=test['Sex'].replace(gender)

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


train.isna().sum()
test.isna().sum()

train['Embarked'].value_counts()
train['Embarked'].fillna('S')

train = train.dropna()
test = test.dropna()

train.isna().sum()
test.isna().sum()


from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
#모델링
#현재'Sex','Pclass','Age','SibSp','Parch' 칼럼을 반영했을때 정확도가 가장 높았다
X_train = train[['Sex','Pclass','Age','SibSp','Parch']]
y_train = train["Survived"]

X_test = test[['Sex','Pclass','Age','SibSp','Parch']]


lr = LogisticRegression()

dt = DecisionTreeClassifier()


lr.fit(X_train,y_train)

dt.fit(X_train,y_train)

lr.predict(X_test)
dt.predict(X_test)
lr_pred=lr.predict_proba(X_test)[:,1]
dt_pred=dt.predict_proba(X_test)[:,1]
print(lr.score(X_train,y_train))
print(dt.score(X_train,y_train))
#출력

submission["Survived"] = lr_pred
submission.to_csv('logistic_regression_pred.csv', index =False)
submission["Survived"] = dt_pred
submission.to_csv('decision_tree_pred.csv', index =False)
