#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:51:04 2020

@author: cpprhtn
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = [0,1,2,3,4,5,6,7,8,9]
y1 = []
y2 = []
y3 = []

for i in x:
    y1.append(i)
    y2.append(i**2)
    y3.append(-i**2)
    
data = pd.DataFrame()
data

data['x'] = x
data['y1'] = y1
data['y2'] = y2
data['y3'] = y3
data


plt.plot(data['x'], data['y1'], 'b-')
plt.plot(data['x'], data['y2'], 'r--')
plt.plot(data['x'], data['y3'], 'go-')



plt.plot(data['x'], data['y1'], 'b-')
plt.show()
plt.plot(data['x'], data['y2'], 'r--')
plt.show()
plt.plot(data['x'], data['y3'], 'go-')
plt.show()


#표 추
# nml -> n x m로 분할 후 l번째 칸에 표시
plt.subplot(311)
plt.plot(data['x'], data['y1'], 'b-')
plt.subplot(312)
plt.plot(data['x'], data['y2'], 'r--')
plt.subplot(313)
plt.plot(data['x'], data['y3'], 'go-')


#축 라벨 추
plt.plot(data['x'], data['y1'], 'b-')
plt.plot(data['x'], data['y2'], 'r--')
plt.plot(data['x'], data['y3'], 'go-')

plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

plt.show()


#타이틀 추가
plt.plot(data['x'], data['y1'], 'b-')
plt.plot(data['x'], data['y2'], 'r--')
plt.plot(data['x'], data['y3'], 'go-')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

plt.title('x-y graph', fontsize=20)

plt.show()



#데이터 라벨 추가
plt.plot(data['x'], data['y1'], 'b-', label='y1')
plt.plot(data['x'], data['y2'], 'r--', label='y2')
plt.plot(data['x'], data['y3'], 'go-', label='y3')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('x-y graph', fontsize=15)

plt.legend()

plt.show()


#그래프 크기 지정
plt.figure(figsize=(15,7))
plt.plot(data['x'], data['y1'], 'b-', label='y1')
plt.plot(data['x'], data['y2'], 'r--', label='y2')
plt.plot(data['x'], data['y3'], 'go-', label='y3')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('x-y graph', fontsize=15)
plt.legend()
plt.show()


#수직, 수평선 추가
plt.figure(figsize=(15,7))
plt.plot(data['x'], data['y1'], 'b-', label='y1')
plt.plot(data['x'], data['y2'], 'r--', label='y2')
plt.plot(data['x'], data['y3'], 'go-', label='y3')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('x-y graph', fontsize=15)

plt.axvline(x=4, color='gray')
plt.axhline(y=20, color='gray')

plt.legend()
plt.show()



#그래프 추가
from matplotlib import rc
rc('font', family='AppleGothic')
plt.figure(figsize=(15,7))
plt.plot(data['x'], data['y1'], 'b-', label='y1')
plt.plot(data['x'], data['y2'], 'r--', label='y2')
plt.plot(data['x'], data['y3'], 'go-', label='y3')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('x-y graph', fontsize=15)

plt.axvline(x=4, color='gray')
plt.axhline(y=20, color='gray')

plt.text(1, 30, '안녕', fontsize=30)

plt.legend()
plt.show()


#그리드 추가
plt.figure(figsize=(15,7))
plt.plot(data['x'], data['y1'], 'b-', label='y1')
plt.plot(data['x'], data['y2'], 'r--', label='y2')
plt.plot(data['x'], data['y3'], 'go-', label='y3')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('x-y graph', fontsize=15)
plt.legend()

plt.grid()
plt.show()


#막대그래프
plt.figure(figsize=(15,7))

plt.bar(data['x'], data['y3'], label='y3')
plt.bar(data['x'], data['y2'], label='y2')
plt.bar(data['x'], data['y1'], label='y1')
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('x-y graph', fontsize=15)
plt.legend()
plt.grid()
plt.show()




train = pd.read_csv('/Users/cpprhtn/Desktop/git_local/Pusan_AI_ML/titanic/train.csv')
test = pd.read_csv('/Users/cpprhtn/Desktop/git_local/Pusan_AI_ML/titanic/test.csv')
submission= pd.read_csv("/Users/cpprhtn/Documents/Python_Busan/titanic/sample_submission.csv")



train.isna().sum()
train.median()
train.describe()
train["Age"] = train["Age"].fillna(28)
test["Age"] = test["Age"].fillna(28)
train["Embarked"] = train["Embarked"].fillna("S")



column = 'Pclass'

survived = train[train['Survived']==1]
dead = train[train['Survived']==0]
survived_cnt = survived[column].value_counts()
dead_cnt = dead[column].value_counts()
df = pd.DataFrame([survived_cnt, dead_cnt])
df.index = ['survived', 'dead']
df.plot(kind='bar', figsize=(10,4))
plt.grid()
plt.show()

df = df.T
df['survival_rate'] = 100*df['survived']/(df['survived']+df['dead'])
df['survival_rate'].plot(kind='bar', figsize=(10,4))
plt.show()
df

df['survival_rate']
#column = 'Pclass'
plt.figure(figsize=(10,4))
plt.bar(df.index.astype('str'), df['survival_rate'])
plt.xlabel(column)
plt.ylabel('survival_rate')
plt.title('Pclass-Survival_rate')
plt.show()


train['Fare'].hist(rwidth = 0.8)

train['family'] = train['SibSp'] + train['Parch']
test['family'] = test['SibSp'] + test['Parch']

train.loc[train['Age'] < 10, 'Age_bin'] = 0
train.loc[(train['Age'] >= 10) & (train['Age'] < 20), 'Age_bin'] = 1
train.loc[(train['Age'] >= 20) & (train['Age'] < 30), 'Age_bin'] = 2
train.loc[(train['Age'] >= 30) & (train['Age'] < 40), 'Age_bin'] = 3
train.loc[(train['Age'] >= 40) & (train['Age'] < 50), 'Age_bin'] = 4
train.loc[(train['Age'] >= 50) & (train['Age'] < 60), 'Age_bin'] = 5
train.loc[(train['Age'] >= 60) & (train['Age'] < 70), 'Age_bin'] = 6
train.loc[train['Age'] >= 70, 'Age_bin'] = 7

test.loc[test['Age'] < 10, 'Age_bin'] = 0
test.loc[(test['Age'] >= 10) & (test['Age'] < 20), 'Age_bin'] = 1
test.loc[(test['Age'] >= 20) & (test['Age'] < 30), 'Age_bin'] = 2
test.loc[(test['Age'] >= 30) & (test['Age'] < 40), 'Age_bin'] = 3
test.loc[(test['Age'] >= 40) & (test['Age'] < 50), 'Age_bin'] = 4
test.loc[(test['Age'] >= 50) & (test['Age'] < 60), 'Age_bin'] = 5
test.loc[(test['Age'] >= 60) & (test['Age'] < 70), 'Age_bin'] = 6
test.loc[test['Age'] >= 70, 'Age_bin'] = 7


x_train = train[['Pclass', 'Sex', 'Embarked', 'family', 'Age_bin']]
y_train = train['Survived']

x_test = test[['Pclass', 'Sex', 'Embarked', 'family', 'Age_bin']]

x_train['Sex'] = x_train['Sex'].map({'male':0, 'female':1})
x_test['Sex'] = x_test['Sex'].map({'male':0, 'female':1})

x_train['Embarked'] = x_train['Embarked'].map({'S':0, 'C':1, 'Q':2})
x_test['Embarked'] = x_test['Embarked'].map({'S':0, 'C':1, 'Q':2})



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dt_model = DecisionTreeClassifier(random_state=777)
rf_model_1 = RandomForestClassifier(n_jobs=-1, random_state=777, n_estimators=3)
rf_model_2 = RandomForestClassifier(n_jobs=-1, random_state=777, n_estimators=10)


dt_model.fit(x_train, y_train)
rf_model_1.fit(x_train, y_train)
rf_model_2.fit(x_train, y_train)

submission_dt = submission.copy()
submission_rf_1 = submission.copy()
submission_rf_2 = submission.copy()


#예측결과 기록
submission_dt['Survived'] = dt_model.predict_proba(x_test)[:, 1]
submission_rf_1['Survived'] = rf_model_1.predict_proba(x_test)[:, 1]
submission_rf_2['Survived'] = rf_model_2.predict_proba(x_test)[:, 1]

submission_dt.to_csv('에측 값/submission_dt(2).csv', index=False)
submission_rf_1.to_csv('예측 값/submission_rf_1(2).csv', index=False)
submission_rf_2.to_csv('예측 값/submission_rf_2(2).csv', index=False)


#상관관계 분석
train['Sex'] = train['Sex'].map({'male':0, 'female':1})
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q':2})

plt.figure(figsize=(10,9))
sns.heatmap(train.corr(), annot=True)







