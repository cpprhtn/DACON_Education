#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:11:46 2020

@author: cpprhtn
"""

import pandas as pd
import lightgbm as lgb

train = pd.read_csv("movies_train.csv")
test = pd.read_csv("movies_test.csv")
submission= pd.read_csv("submission.csv")

train.head()


train['dir_prev_bfnum'].fillna(0, inplace=True)
test['dir_prev_bfnum'].fillna(0, inplace=True)

train.isna().sum()
test.isna().sum()




train["genre"]=train["genre"].map({"액션":0,"느와르":1,"코미디":2,"다큐멘터리":3,
                                   "뮤지컬":4,"드라마":5,"멜로/로맨스":6,"공포":7,"애니메이션":8,
                                   "SF":9,"미스터리":10,"서스펜스":11})

test["genre"]=test["genre"].map({"액션":0,"느와르":1,"코미디":2,"다큐멘터리":3,
                                   "뮤지컬":4,"드라마":5,"멜로/로맨스":6,"공포":7,"애니메이션":8,
                                   "SF":9,"미스터리":10,"서스펜스":11})


model = lgb.LGBMRegressor(n_estimators=1000)

features = ['genre','time','num_staff','num_actor','dir_prev_num','dir_prev_bfnum']
traget=['box_off_num']

X_train, X_test, y_train = train[features], test[features], train[traget]

model.fit(X_train,y_train)


singleLGBM = submission.copy()
singleLGBM.head()
singleLGBM['box_off_num']=model.predict(X_test)
singleLGBM.to_csv('singleLGBM1.csv',index=False)


from sklearn.model_selection import KFold

k_fold = KFold(n_splits=5,shuffle=True)

model = lgb.LGBMRegressor(n_estimators=1000)

models = []

for train_idx,val_idx in k_fold.split(X_train):
    x_t = X_train.iloc[train_idx]
    y_t = y_train.iloc[train_idx]
    x_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]
    models.append(model.fit(x_t, y_t, eval_set=(x_val,y_val), early_stopping_rounds= 100, verbose=100))
 

preds = []
for model in models:
    preds.append(model.predict(X_test))
    
len(preds)
    
kfoldLightGBM1["box_off_num"]
kfoldLightGBM1 = submission.copy()

import numpy as np    

kfoldLightGBM1['box_off_num'] = np.mean(preds, axis=0)

kfoldLightGBM1.to_csv('kfoldLGBM1.csv',index=False)

for i in len(kfoldLightGBM1):
    if kfoldLightGBM1["box_off_num"][i] < 0:
        
kfoldLightGBM1["box_off_num"][i]=kfoldLightGBM1["box_off_num"][i].map({"액션":0,"느와르":1})
