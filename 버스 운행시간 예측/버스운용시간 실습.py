#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:19:54 2020

@author: cpprhtn
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("submission_제출양식.csv")

train.isna().sum()
train.info()

#datetime로 형변환
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

train.info()
train.groupby(['date'])['next_arrive_time'].mean().plot(kind='bar')

#요일 데이터로 변환
#0-월 1-화 ~~~
train['weekday'] = train['date'].dt.weekday
test['weekday'] = test['date'].dt.weekday

train
#요일별 평균시간
train.groupby(['weekday'])['next_arrive_time'].mean().plot() #5,6 -> 즉 주말에서 평균 시간이 크게 하락

#노선별 한정거장 이동시
train['route_nm'].unique()
test['route_nm'].unique()

train['route_nm'].unique() == test['route_nm'].unique()


train.groupby(['route_nm'])['next_arrive_time'].mean().plot(kind='bar')


#버스 노선 id를 숫자로 라벨링
route_label = {}
for idx, route in enumerate(train['route_nm'].unique()):
    route_label[route] = idx


route_label

train['route_nm'] = train['route_nm'].map(route_label)
test['route_nm'] = test['route_nm'].map(route_label)

train.info()


#정류소 이름 숫자로 labeling
#pd.concat() : 데이터프레임들을 위아래 또는 좌우로 붙혀 하나로 합침



#data값은 따로 존재하진 않음, 그냥 전체 길이 확인 가능
unique_station = pd.concat([train['now_station'], train['next_station'], test['now_station'], test['next_station']], axis=0).unique()

len(unique_station)


station_label = {}
for idx, station in enumerate(unique_station):
    station_label[station] = idx

station_label
    
train['now_station'] = train['now_station'].map(station_label)
train['next_station'] = train['next_station'].map(station_label)
test['now_station'] = test['now_station'].map(station_label)
test['next_station'] = test['next_station'].map(station_label)
    
# 시간별 다음 정류소까지 걸리는 평균 시간





train.groupby(['now_arrive_time'])['next_arrive_time'].mean().plot(kind='bar')
#공통적인 맨 뒤의 ~시 부분을 짤라줌
train['now_arrive_time'] = train['now_arrive_time'].str[:-1].astype(int)
test['now_arrive_time'] = test['now_arrive_time'].str[:-1].astype(int)


sns.lmplot('distance', 'next_arrive_time', data = train, fit_reg=True)


import folium
from folium.plugins import MarkerCluster

jeju=(33.51411, 126.52969) # 제주시 근처

map_osm= folium.Map((33.399835, 126.506031),zoom_start=10)
mc = MarkerCluster()

mc.add_child( folium.Marker(location=jeju,popup='제주시',icon=folium.Icon(color='red',icon='info-sign') ) )
map_osm.add_child(mc)

sns.lmplot('now_longitude', 'now_latitude', data = train, fit_reg=False)
plt.title('station')
plt.show()




from sklearn.cluster import KMeans

kmeans_clst = KMeans(n_clusters=10).fit(train[['now_latitude', 'now_longitude']])
train['now_clst'] = kmeans_clst.labels_

sns.lmplot('now_longitude', 'now_latitude', data = train, fit_reg=False, hue="now_clst")
plt.title('now_station')
plt.show()

#클러스터링 구간별 다음역 도착 평균시간
train.groupby(['now_clst'])['next_arrive_time'].mean().plot(kind='bar')


#구간 라벨링
now_station = {}
for i in range(10): # 총 10개 구간으로 클러스터링
    for station in train[train['now_clst']==i]['now_station'].unique(): # 각 구간의 정류소
        now_station[station] = i # 해당 정류소의 label

now_station
test['now_clst'] = test['now_station'].map(now_station) # test도 train과 동일하게 labeling


sns.lmplot('now_longitude', 'now_latitude', data = test, fit_reg=False, hue="now_clst")
plt.title('now_station')
plt.show()




kmeans_clst = KMeans(n_clusters=10).fit(train[['next_latitude', 'next_longitude']])
train['next_clst'] = kmeans_clst.labels_

next_station = {}
for i in range(10):
    for station in train[train['next_clst']==i]['next_station'].unique():
        next_station[station] = i

test['next_clst'] = test['next_station'].map(next_station)

sns.lmplot('next_longitude', 'next_latitude', data = train, fit_reg=False, hue="next_clst")
plt.title('next_station')
plt.show()

#구간별 전 정류소에서 오는 시간
train.groupby(['next_clst'])['next_arrive_time'].mean().plot(kind='bar')


#특성
columns = ['route_nm', 'now_station', 'now_arrive_time', 'distance', 'next_station', 'now_clst', 'next_clst']
X_train = train[columns]
Y_train = train['next_arrive_time']
x_test = test[columns]



from sklearn.model_selection import KFold
'''
import xgboost as xgb


k_fold = KFold(n_splits=5, shuffle=True, random_state=777)

xgb_models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_train, y_train = X_train.iloc[train_idx], Y_train.iloc[train_idx]
    x_val, y_val = X_train.iloc[val_idx], Y_train.iloc[val_idx]
    
    d_train = xgb.DMatrix(x_train, y_train)
    d_val = xgb.DMatrix(x_val, y_val)
    watchlist = [(d_train, 'train'), (d_val, 'eval')]
    
    params = {
            'seed':777
            }
    
    xgb_models.append(xgb.train(params, d_train, 1000, watchlist, verbose_eval=100, early_stopping_rounds=100))
'''


import lightgbm as lgb

k_fold = KFold(n_splits=5, shuffle=True)

lgb_models = []

for train_idx, val_idx in k_fold.split(X_train):
    x_train, y_train = X_train.iloc[train_idx], Y_train.iloc[train_idx]
    x_val, y_val = X_train.iloc[val_idx], Y_train.iloc[val_idx]
    
    d_train = lgb.Dataset(x_train, y_train)
    d_val = lgb.Dataset(x_val, y_val)

    params = {
            'objective': 'regression', # regression, binary, multiclass
            'metric':'rmse'
            }
    
    lgb_models.append(lgb.train(params, d_train, 1000, d_val,verbose_eval=100, early_stopping_rounds=100))



submission_lgb = submission.copy()
preds = []
for model in lgb_models:
    preds.append(model.predict(x_test))
len(preds)

pred = (preds[0]+preds[1]+preds[2]+preds[3]+preds[4])/5
submission_lgb['next_arrive_time'] = pred
submission_lgb

submission_lgb.to_csv('submission_lgb2.csv', index=False)






