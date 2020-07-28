# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:03:23 2020

@author: cpprh
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission= pd.read_csv("submission_sample.csv")

#데이터 합치기 
train_len = train.shape[0]
merged = train.append(test, ignore_index=True) 
print("train len : ", train.shape[0])
print("test len : ", test.shape[0])
print("merged len : ", merged.shape[0])

merged.isna().sum()
#datetime로 형변환
merged['date2'] = pd.to_datetime(train['date'])
merged['weekday'] = merged['date2'].dt.weekday
gender ={'시내':0,'시외':1}
merged['in_out']=merged['in_out'].replace(gender)


#버스노선에따른 stop 수
bus_route = merged.groupby(['bus_route_id'])['station_code'].unique().reset_index(name='stop_number')
bus_route['stop_number'] = bus_route['stop_number'].apply(lambda x : x.tolist())
bus_route['cnt'] = bus_route['stop_number'].apply(lambda x : len(x))
bus_route = bus_route.sort_values(by='cnt', ascending=False).reset_index(drop=True)
plt.title("Number of stops by bus route")
plt.plot(bus_route.index, bus_route['cnt'])
plt.show()

merged = pd.get_dummies(merged,columns=['weekday'])
merged = pd.get_dummies(merged,columns=['in_out'])

from sklearn.cluster import KMeans
kmeans_clst = KMeans(n_clusters=20).fit(merged[['bus_route_id']])
merged['bus_route_clst'] = kmeans_clst.labels_


merged['6_8_in']=merged['6~7_ride']+merged['7~8_ride'] 
merged['8_10_in']=merged['8~9_ride']+merged['9~10_ride']
merged['10_12_in']=merged['10~11_ride']+merged['11~12_ride']
del merged['6~7_ride']
del merged['7~8_ride']
del merged['8~9_ride']
del merged['9~10_ride']
del merged['10~11_ride']
del merged['11~12_ride']

merged['6_8_out']=merged['6~7_takeoff']+merged['7~8_takeoff'] 
merged['8_10_out']=merged['8~9_takeoff']+merged['9~10_takeoff']
merged['10_12_out']=merged['10~11_takeoff']+merged['11~12_takeoff']
del merged['6~7_takeoff']
del merged['7~8_takeoff']
del merged['8~9_takeoff']
del merged['9~10_takeoff']
del merged['10~11_takeoff']
del merged['11~12_takeoff']


merged['6_8_move'] = merged['6_8_in'] - merged['6_8_out']
merged['8_10_move'] = merged['8_10_in'] - merged['8_10_out']
merged['10_12_move'] = merged['10_12_in'] - merged['10_12_out']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
merged[ = scaler.fit_transform(merged)
#데이터 나누기
train = merged[:train_len]
test = merged[train_len:]
train.isna().sum()

columns = ['bus_route_id','station_code',
           '6_8_in','8_10_in','10_12_in', '10_12_move', '6_8_move', '8_10_move', '6_8_out', '8_10_out',
           'weekday_0.0','weekday_1.0','weekday_2.0','weekday_3.0','weekday_4.0','weekday_5.0','weekday_6.0',
           'in_out_0','in_out_1', ]
X_train = train[columns]
Y_train = train['18~20_ride']
x_test = test[columns]
'''
#정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
merged = scaler.fit_transform(merged)
'''

from sklearn.model_selection import KFold
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
submission_lgb['18~20_ride'] = pred
submission_lgb

submission_lgb.to_csv('submission_lgb6.csv', index=False)


