#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:06:04 2020

@author: cpprhtn
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(777)
data = np.random.rand(100,2)
data = pd.DataFrame(data)
data.columns = ['x', 'y']
data

sns.lmplot('x', 'y', data = data, fit_reg=False)


from sklearn.cluster import KMeans

kmeans_clst = KMeans(n_clusters=10, random_state=777).fit(data[['x', 'y']])
data['clst'] = kmeans_clst.labels_

sns.lmplot('x', 'y', data = data, fit_reg=False, hue="clst")
