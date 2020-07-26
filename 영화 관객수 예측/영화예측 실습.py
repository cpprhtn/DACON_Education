#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:11:46 2020

@author: cpprhtn
"""


'''
Test.verson
파라미터값 찾아보기
'''
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


train['distributor'] = train['distributor'].apply(merge_distributor)
test['distributor'] = test['distributor'].apply(merge_distributor)

movie_score = {'CJ 엔터테인먼트' : 1,
               '롯데엔터테인먼트' : 2,
               '(주)NEW' : 3,
               'CGV아트하우스' : 4,
               'NEW' : 5,
               '필라멘트 픽쳐스' :6 ,
               '이십세기폭스코리아(주)' : 7,
              '(주)쇼박스' : 8}

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
    

kfoldLightGBM1 = submission.copy()

import numpy as np    

kfoldLightGBM1['box_off_num'] = np.mean(preds, axis=0)

kfoldLightGBM1.to_csv('kfoldLGBM1.csv',index=False)







train_label = train['box_off_num']
train_data = train.drop(['title','director', 'box_off_num'], axis=1)
test_data = test.drop(['title','director'], axis=1)

kfolds = KFold(n_splits=5, shuffle=True)

#교차 검증을 위해
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(estimator=model, X=train_data, y=train_label, scoring='neg_mean_squared_error', cv=kfolds))
    return rmse




param = {
    'num_leaves' : 31,
    'min_data_in_leaf' : 10,
    'objective' : 'regression',
    'max_depth' : -1,
    'learning_rate' : 0.001,
    'min_child_samples' : 10,
    'feature_fraction' : 0.9,
    'metric' : 'rmse', #채점방식과 동일
    'verbosity' : -1,
    'nthread' : -1,
}


folds = KFold(n_splits=5, shuffle=True)
oof1 = np.zeros(len(train_data))
predictions = np.zeros(len(test_data))
feature_importance_df = pd.DataFrame()


for fold_, (trn_index, val_index) in enumerate(folds.split(train_data)):
    trn_data = lgb.Dataset(train_data.iloc[trn_index], label=train_label.iloc[trn_index])
    val_data = lgb.Dataset(train_data.iloc[val_index], label=train_label.iloc[val_index])
    
    num_round = 5000
    clf = lgb.train(params=param, train_set=trn_data, num_boost_round=num_round, valid_sets=[trn_data, val_data], verbose_eval=500, early_stopping_rounds=500)
    oof1[val_index] = clf.predict(train_data.iloc[val_index], num_iterations=clf.best_iteration) # 나중에 스태킹을 위한 train 데이터도 된다.
    # feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df['Feature'] = train_data.columns
    fold_importance_df['importance'] = clf.feature_importance()
    
    fold_importance_df['fold'] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_data, num_iterations=clf.best_iteration) / folds.n_splits # 나중에 스태킹을 위한 test 데이터도 된다.
    

