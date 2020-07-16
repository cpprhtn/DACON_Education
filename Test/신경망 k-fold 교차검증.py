#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:38:28 2020

@author: cpprhtn
"""



import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# 랜덤 시드를 설정합니다.
np.random.seed(0)

# 특성 개수
number_of_features = 100

# 특성 행렬과 타깃 벡터를 만듭니다.
features, target = make_classification(n_samples = 10000,
                                       n_features = number_of_features,
                                       n_informative = 3,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.5, .5],
                                       random_state = 0)

# 설정 완료된 신경망을 반환하는 함수를 만듭니다.
def create_network():

    # 신경망 모델을 만듭니다.
    network = models.Sequential()
    
    network.add(layers.Dropout(0.2, input_shape=(number_of_features,)))
    # 렐루 활성화 함수를 사용한 완전 연결 층을 추가합니다.
    network.add(layers.Dense(units=16, activation="relu"))
    
    network.add(layers.Dropout(0.5))

    # 렐루 활성화 함수를 사용한 완전 연결 층을 추가합니다.
    network.add(layers.Dense(units=16, activation="relu"))
    
    network.add(layers.Dropout(0.5))

    # 시그모이드 활성화 함수를 사용한 완전 연결 층을 추가합니다.
    network.add(layers.Dense(units=1, activation="sigmoid"))

    # 신경망의 모델 설정을 완료합니다.
    network.compile(loss="binary_crossentropy", # 크로스 엔트로피
                    optimizer="rmsprop", # 옵티마이저
                    metrics=["accuracy"]) # 성능 지표

    # 설정 완료된 모델을 반환합니다.
    return network

# 케라스 모델을 래핑하여 사이킷런에서 사용할 수 있도록 만듭니다.
neural_network = KerasClassifier(build_fn=create_network,
                                 epochs=10,
                                 batch_size=100,
                                 verbose=1)

# 3-폴드 교차검증을 사용하여 신경망을 평가합니다.
cross_val_score(neural_network, features, target, cv=3)
