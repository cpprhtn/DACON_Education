#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:30:21 2020

@author: cpprhtn
"""


#Logistic Regression 최종출력값 = 0 또는 1

1. 데이터 전처리
gender = { "male":0, "female":1}
merged['Sex']=merged['Sex'].replace(gender)

merged['Age'].fillna(merged.groupby('Pclass')['Age'].transform('mean'), inplace=True)

train['date'] = pd.to_datetime(train['date']).dt.weekday

#요일별 평균시간
train.groupby(['weekday'])['next_arrive_time'].mean().plot()

.unique()

2. 데이터셋 분리

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2)

3. 데이터 정규화(스케일링)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

4. 모델 생성

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train_features, train_labels)
 
model.coef_ #어떤 feature가 결과에 영향을 미치는지 확인

5. test 데이터 적용
pred = scaler.transform(test)
model.predict(pred)


