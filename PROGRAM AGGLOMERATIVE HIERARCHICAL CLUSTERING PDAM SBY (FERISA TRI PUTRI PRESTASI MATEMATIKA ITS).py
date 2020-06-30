# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 00:58:01 2019

@author: Ferisa
"""

import pandas as pd
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

train = pd.read_csv('Book1.csv')
X = train.iloc[:, [0, 1]].values
Y = train.iloc[:,3].values
plt.scatter(X[Y == 502021, 0], X[Y == 502021, 1], s = 100, c = 'red', label = 'Bendel 50202A')
plt.scatter(X[Y == 502471, 0], X[Y == 502471, 1], s = 100, c = 'green', label = 'Bendel 50247A')
plt.xlabel('Latitude', fontsize=18)
plt.ylabel('Longitude', fontsize=18)
plt.grid()
plt.show()


import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()

from sklearn.cluster import AgglomerativeClustering
function = AgglomerativeClustering(linkage='ward', 
                              affinity='euclidean', 
                              n_clusters=99)

hasilcluster  = function.fit_predict(X)


plt.scatter(X[hasilcluster == 0, 0], X[hasilcluster == 0, 1], s = 100, c = 'red', label = '50202A')
plt.scatter(X[hasilcluster == 1, 0], X[hasilcluster == 1, 1], s = 100, c = 'blue', label = '50247A')

plt.title('CLUSTER DATA PELANGGAN PDAM')
plt.xlabel('POS_LAT')
plt.ylabel('POS_LONG')
plt.legend()
plt.show()
