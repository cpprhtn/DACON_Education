#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:59:13 2020

@author: cpprhtn
"""


import pandas as pd

train = pd.read_csv("/Users/cpprhtn/Documents/Python_Busan/titanic/train.csv")
test = pd.read_csv("/Users/cpprhtn/Documents/Python_Busan/titanic/test.csv")
submission= pd.read_csv("/Users/cpprhtn/Documents/Python_Busan/titanic/sample_submission.csv")
#전처리
train["Age"] = train["Age"].fillna(28)
test["Age"] = test["Age"].fillna(28)
train["Embarked"] = train["Embarked"].fillna("S")
train.isna().sum()
test.isna().sum()
train["Sex"]=train["Sex"].map({"male":0,"female":1})

''' 사실상 불필요
survived = train[train["Survived"]==0]
dead = train[train["Survived"]==1]
survived_cnt=survived["Pclass"].value_counts()
dead_cnt=dead["Pclass"].value_counts()

df = pd.DataFrame([survived_cnt,dead_cnt])
df.index=["survived","dead"]

df.plot(kind="bar",figsize=(15,8))
df["survived_rate"].plot(kind="bar",figsize=(15,8))

df= df.T
df["survived_rate"]=100*df["survived"]/(df["survived"]+df["dead"])
'''


from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
#모델링
X_train = train[['Sex','Pclass','Age']]
y_train = train["Survived"]

test = test[['Sex','Pclass','Age']]
test["Sex"]=test["Sex"].map({"male":0,"female":1})
X_test = test

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
