#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:06:03 2020

@author: cpprhtn
"""



import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

train = pd.read_csv("따릉이/train.csv")
test = pd.read_csv("따릉이/test.csv")
submission= pd.read_csv("따릉이/submission.csv")




train[train['hour_bef_temperature'].isna()]
test[test['hour_bef_temperature'].isna()]
train.groupby('hour').mean()['hour_bef_temperature']
train['hour_bef_temperature'].fillna({934:14.788136, 1035:20.926667},inplace = True)
train['hour_bef_windspeed'].fillna({18:3.281356, 244:1.836667, 260:1.9655517, 376:1.965517, 780:3.278333, 934:1.965517, 1035:3.838333, 1138:2.766667, 1229:1.633333},inplace = True)
test['hour_bef_temperature'].fillna(19.704918, inplace = True)
test['hour_bef_windspeed'].fillna(3.595072, inplace = True)

train[train['hour_bef_temperature'].isna()].index
train[train['hour_bef_windspeed'].isna()].index

test.isna().sum()
test["hour_bef_precipitation"] = test["hour_bef_precipitation"].fillna(0)
train["hour_bef_precipitation"] = train["hour_bef_precipitation"].fillna(0)
test.mean()
#test["hour_bef_humidity"] = test["hour_bef_humidity"].fillna(56.588811)
#train["hour_bef_humidity"] = train["hour_bef_humidity"].fillna(56.588811)
test["hour_bef_humidity"] = test["hour_bef_humidity"].fillna(0)
train["hour_bef_humidity"] = train["hour_bef_humidity"].fillna(0)
train.isna().sum()

features = ['hour','hour_bef_temperature','hour_bef_windspeed','hour_bef_precipitation','hour_bef_humidity']
X_train = train[features]
y_train = train['count']
X_test = test[features]

model100 =  RandomForestRegressor(n_estimators=5000, random_state = 0)
#model100_5 =  RandomForestRegressor(n_estimators=100, max_depth = 5, random_state=0)
#model200 =  RandomForestRegressor(n_estimators=200)

model100.fit(X_train, y_train)
#model100_5.fit(X_train, y_train)
#model200.fit(X_train, y_train)

ypred1 = model100.predict(X_test)
#ypred2 = model100_5.predict(X_test)
#ypred3 = model200.predict(X_test)

submission['count'] = ypred1
submission.to_csv('model100.csv', index=False)
#submission['count'] = ypred2
#submission.to_csv('model100_5.csv', index=False)
#submission['count'] = ypred1
#submission.to_csv('model200.csv', index=False)




