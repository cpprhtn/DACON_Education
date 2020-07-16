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

'''
전처리
'''

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

#Parch와 SibSp 칼럼을 Family칼럼으로 합침
merged[Columns.Family] = merged[Columns.Parch] + merged[Columns.SibSp] + 1 #자기자신
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

#안쓰는 데이터 drop
merged = merged.drop([Columns.Name,Columns.Ticket,Columns.Cabin], axis=1) 


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
    

merged[Columns.Fare] = merged[Columns.Fare].map(lambda i : np.log(i) if i > 0 else 0)

sns.distplot(merged[Columns.Fare])


'''
불필요한 열 삭제
'''
if Columns.Ticket in merged:
    merged = merged.drop(labels=[Columns.Ticket], axis=1)
if Columns.Cabin in merged:
    merged = merged.drop(labels=[Columns.Cabin], axis=1)
if Columns.Deck in merged:
    merged = merged.drop(labels=[Columns.Deck], axis=1)


merged.describe(include='all')


#카테고리 데이터는 one-hot 인코딩으로 변경한다.
merged.head()
#Pclass
#Embarked
merged = pd.get_dummies(merged, columns=[Columns.Pclass], prefix='Pclass')
merged = pd.get_dummies(merged, columns=[Columns.Embarked], prefix='Embarked')
merged = pd.get_dummies(merged, columns=[Columns.Sex], prefix='Sex')

merged.head()


'''
수치 데이터는 일정 값으로 scaling 해주는 것이 필요하다.
입력 데이터를 0~1로 scaling하는 함수
'''
from sklearn.preprocessing import MinMaxScaler

class NoColumnError(Exception):
    """Raised when no column in dataframe"""
    def __init__(self, value):
        self.value = value
    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.value))

# normalize AgeGroup
def normalize_column(data, columnName):
    scaler = MinMaxScaler(feature_range=(0, 1))    
    if columnName in data:
        aaa = scaler.fit_transform(data[columnName].values.reshape(-1, 1)) # 입력을 2D 데이터로 넣어야 하므로 reshape해 준다.
        aaa = aaa.reshape(-1,) # 다시 원복해서 넣어주지만, 그냥 넣어도 알아서 제대로 들어간다...
        #print(aaa.shape)
        data[columnName] = aaa
        return data
    else:
        raise NoColumnError(str(columnName) + " is not exists!")

def normalize(dataset, columns):
    for col in columns:
        dataset = normalize_column(dataset, col)
    return dataset


#Age, Fare, Family 칼럼 스케일링
merged = normalize(merged, [Columns.Age, Columns.Fare, Columns.Family])

merged.head(n=10)

'''
merged를 train/test로 분리한다.
'''

train = merged[:train_len]

test = merged[train_len:]


train.isna().sum()

train = train.drop([Columns.PassengerId], axis=1)


test.isna().sum()

test = test.drop([Columns.Survived], axis=1)



test_passenger_id = test[Columns.PassengerId]
test = test.drop([Columns.PassengerId], axis=1)

print(train.shape)
print(test.shape)

train_X = train.drop([Columns.Survived], axis=1).values #Series.values는 numpy array 타입의 데이터임
train_Y = train[Columns.Survived].values.reshape(-1, 1)
print(train_X.shape)
print(train_Y.shape)



'''
keras로 FC레이어를 몇 개 쌓아서 만든 model 생성
kfold cross-validation 사용해서 model train
2에서 train완료된 모델 중 accuracy가 가장 높은 모델 선택해서 test데이터 판정
csv파일로 생성
'''
import keras
from keras import layers, models
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


train_X = train_X.astype(np.float32)
train_Y = train_Y.astype(np.float32)


class SimpleNN(Sequential):
    def __init__(self, input_shape, dropout):
        super().__init__()
        
        self.add(layers.Dense(units=255, activation='relu', input_shape=input_shape))
        self.add(layers.Dropout(dropout))
        
        self.add(layers.Dense(units=8, activation='relu'))
        self.add(layers.Dropout(dropout))
        
        self.add(layers.Dense(units=1, activation='sigmoid'))
        self.add(layers.Dropout(dropout))
        
        self.compile(loss="mse", optimizer='adam', metrics=['accuracy'])



batch_size = 32
epochs = 30
dropout = 0.0
n_splits = 10


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=7)

cvscores = []
models = []
acc_hist = []

for _train, _test in kfold.split(train_X, train_Y):
    model = SimpleNN(input_shape=(train_X.shape[1],), dropout=dropout)
    history = model.fit(train_X[_train], train_Y[_train], epochs=epochs, batch_size=batch_size, verbose=1)
    acc_hist.append(history.history['accuracy'])
    #evaluate model:
    scores = model.evaluate(train_X[_test], train_Y[_test], verbose=0)    
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    models.append(model)

fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(n_splits, 8 * n_splits))
for i in range(len(acc_hist)):
    title = "model " + str(i)    
    ax[i].plot(acc_hist[i])
    ax[i].set_title(title)
plt.show()

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
maxidx = np.argmax(cvscores)
print("Max : [", maxidx, "] : ", "{0:.2f}".format(cvscores[maxidx]))


# 가장 점수가 높은 모델로 test 데이터를 돌린다.
pred = models[maxidx].predict(test, batch_size=test.shape[0], verbose=1)
pred

from sklearn.preprocessing import Binarizer
binarizer=Binarizer(0.6)

test_predict_result=binarizer.fit_transform(pred)
test_predict_result=test_predict_result.astype(np.int32)
#print(test_predict_result[:10])
submission = pd.DataFrame({"PassengerId" : test_passenger_id, "Survived":test_predict_result.reshape(-1)})
submission.to_csv('submission.csv', index=False)






