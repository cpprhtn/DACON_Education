# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:21:07 2020

@author: cpprh
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("sample_submission.csv")


#train/test를 따로 하지 말고, 합쳐서 처리하고 마지막(모델 넣기 직전)에 다시 분리한다.
train_len = train.shape[0]
merged = train.append(test, ignore_index=True) 
print("train len : ", train.shape[0])
print("test len : ", test.shape[0])
print("merged len : ", merged.shape[0])

class Columns:
    # 원래 존재하는 항목
    PassengerId = "PassengerId"
    Survived = "Survived"
    Pclass = "Pclass"
    Name = "Name"
    Sex = "Sex"
    Age = "Age"
    SibSp = "SibSp"
    Parch = "Parch"
    Ticket = "Ticket"
    Fare = "Fare"
    Cabin = "Cabin"
    Embarked = "Embarked"
    # 새로 생성하는 항목
    Title = "Title"
    FareBand = "FareBand"
    Family = "Family"
    Deck = "Deck" # Cabin의 알파벳을 떼서 Deck을 지정한다.
    CabinExists = "CabinExists"

merged[Columns.Family] = merged[Columns.Parch] + merged[Columns.SibSp] + 1
if Columns.Parch in merged:    
    merged = merged.drop([Columns.Parch], axis=1)
if Columns.SibSp in merged:
    merged = merged.drop([Columns.SibSp], axis=1)
    
merged.head()
    

#빈 갯수가 몇개 없으므로 그냥 가장 많은 것으로 채운다.
most_embarked_label = merged[Columns.Embarked].value_counts().index[0]

merged = merged.fillna({Columns.Embarked : most_embarked_label})


gender = { "male":0, "female":1}
merged['Sex']=merged['Sex'].replace(gender)
 
merged.info()   

merged = merged.drop([Columns.Name,Columns.Ticket,Columns.Cabin,Columns.Embarked], axis=1) 


merged.isna().sum()
'''#나이 예측
knn = KNeighborsRegressor()
# 나이가 있는 데이터로 fit해서 모델을 생성
knn.fit(merged[merged['Age'].isnull()==False][merged.columns.drop('Age')],
       merged[merged['Age'].isnull()==False]['Age'])
# 나이가 결측인 데이터를 예측
guesses = knn.predict(merged[merged['Age'].isnull()==True][merged.columns.drop('Age')])
guesses

train.loc[train['Age'].isnull()==True,'Age'] = guesses
'''
merged['Age'].fillna(merged.groupby('Pclass')['Age'].transform('mean'), inplace=True) 
    
merged.loc[merged[Columns.Fare].isnull(), [Columns.Fare]] = merged[Columns.Fare].mean()

sns.distplot(merged[Columns.Fare])
    
    
    
    
    
    
    
    
    
    