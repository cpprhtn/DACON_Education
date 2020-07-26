# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:50:55 2020

@author: cpprh
"""


import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("sample_submission.csv")

train.isna().sum()

train['Age'].fillna(train.groupby('Survived')['Age'].transform('mean'), inplace=True)
train.isna().sum()
train["Age"].mean() #29.750072129886956

test["Age"].fillna(train.groupby('Survived')['Age'].transform('mean'), inplace=True)
train["Embarked"] = train["Embarked"].fillna("S")
train.isna().sum()

train["Sex"]=train["Sex"].replace(["male","female"],[0,1])
test["Sex"]=test["Sex"].replace(["male","female"],[0,1])

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, Y)
X_train = train[['Sex','Pclass','Age','SibSp','Parch']]
y_train = train["Survived"]
X_test = test[['Sex','Pclass','Age','SibSp','Parch']]


from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train)
X_train

X_test = scaler.fit_transform(X_test)
X_test




lr = LogisticRegression()

dt = DecisionTreeClassifier()


lr.fit(X_train,y_train)

dt.fit(X_train,y_train)

train.isna().sum()
test.isna().sum()
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