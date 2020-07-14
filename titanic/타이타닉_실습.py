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
train.info()
train.isna().sum()

df = train.copy()
df2 = train.copy()
gender = { "male":0, "female":1}
df2["Sex"].replace(gender)
df['Age'].fillna(df.groupby('Sex')['Age'].transform('mean'), inplace=True)
df.isna().sum()
train["Age"].mean() #29.69911764705882
df["Age"].mean()

df.describe()
train["Cabin"] #객실 번호

train.groupby(["Embarked"])["Embarked"].plot(kind="bar")
train["Embarked"]

train["Age"] = train["Age"].fillna(30)
test["Age"] = test["Age"].fillna(30)
train["Embarked"] = train["Embarked"].fillna("S")
train.isna().sum()

test["Fare"].mean()
test.isna().sum()

test["Fare"] = test["Fare"].fillna(36)
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
#현재'Sex','Pclass','Age','SibSp','Parch' 칼럼을 반영했을때 정확도가 가장 높았다
X_train = train[['Sex','Pclass','Age','SibSp','Parch','Fare']]
y_train = train["Survived"]

test = test[['Sex','Pclass','Age','SibSp','Parch','Fare']]
test["Sex"]=test["Sex"].map({"male":0,"female":1})
X_test = test

lr = LogisticRegression()

dt = DecisionTreeClassifier()


lr.fit(X_train,y_train)

dt.fit(X_train,y_train)

train.isna().sum()

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
