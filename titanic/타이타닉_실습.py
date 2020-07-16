#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:59:13 2020

@author: cpprhtn
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("sample_submission.csv")



#전처리
train.isna().sum()
test.isna().sum()
'''
train['Age'].fillna(train.groupby('Pclass')['Age'].transform('mean'), inplace=True)
test['Age'].fillna(test.groupby('Pclass')['Age'].transform('mean'), inplace=True)
train['Age'].mean()
'''

train = train[['Sex','Pclass','Age','SibSp','Parch','Survived']]
test = test[['Sex','Pclass','Age','SibSp','Parch']]

gender = { "male":0, "female":1}
train['Sex']=train['Sex'].replace(gender)
test['Sex']=test['Sex'].replace(gender)


knn = KNeighborsRegressor()
# 나이가 있는 데이터로 fit해서 모델을 생성
knn.fit(train[train['Age'].isnull()==False][train.columns.drop('Age')],
       train[train['Age'].isnull()==False]['Age'])
# 나이가 결측인 데이터를 예측
guesses = knn.predict(train[train['Age'].isnull()==True][train.columns.drop('Age')])
guesses

train.loc[train['Age'].isnull()==True,'Age'] = guesses


train.isna().sum()
test.isna().sum() #test 데이터는 나중에 Pre_test로 씀

'''
train = train.dropna()
test = test.dropna()
train.isna().sum()
test.isna().sum()
'''



#현재'Sex','Pclass','Age','SibSp','Parch' 칼럼을 반영했을때 정확도가 가장 높았다
#필요한 칼럼만
X_train = train[['Sex','Pclass','Age','SibSp','Parch']]
y_train = train["Survived"]



from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense

#Train data만 가지고 나눠서 훈련 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=7)

np.random.seed(7)



model = Sequential()
model.add(Dense(255, input_shape=(5,), activation='relu'))
model.add(Dense((1), activation='sigmoid'))
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
model.summary()



hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500)

plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['loss','val_loss', 'accuracy','val_accuracy'])
plt.show()


#예측해야하는 데이터...
Pre_test = test[['Sex','Pclass','Age','SibSp','Parch']]

model.predict(Pre_test)
model_pred=model.predict(Pre_test)

from sklearn.preprocessing import Binarizer
binarizer=Binarizer(0.5)

test_predict_result=binarizer.fit_transform(model_pred)
test_predict_result=test_predict_result.astype(np.int32)
#print(test_predict_result[:10])
submission = pd.DataFrame({"PassengerId" : test_passenger_id, "Survived":test_predict_result.reshape(-1)})
submission.to_csv('submission.csv', index=False)



from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression()

dt = DecisionTreeClassifier()


lr.fit(X_train,y_train)

dt.fit(X_train,y_train)

lr.predict(Pre_test)
dt.predict(Pre_test)
lr_pred=lr.predict_proba(Pre_test)[:,1]
dt_pred=dt.predict_proba(Pre_test)[:,1]
print(lr.score(X_train,y_train))
print(dt.score(X_train,y_train))
#출력

submission["Survived"] = lr_pred
submission.to_csv('logistic_regression_pred.csv', index =False)
submission["Survived"] = dt_pred
submission.to_csv('decision_tree_pred.csv', index =False)
'''