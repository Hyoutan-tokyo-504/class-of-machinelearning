#!/usr/bin/env python
# coding: utf-8

# # PCA実装

# In[1]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def cov_matrix(X):
    # X: 行列（2次元配列）
    matrix = np.zeros((X.shape[1],X.shape[1]))
    ### 入力の行列の列間の共分散行列を計算するコード
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            matrix[i][j] = np.dot(X[:,i]-np.mean(X[:,i]),X[:,j]-np.mean(X[:,j]))/X.shape[0]
    return matrix


# In[6]:


def pca(X, k):
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
    matrix = cov_matrix(X)
    ### 3. 共分散行列の固有値・固有ベクトルを計算するコード
    w, v = np.linalg.eig(matrix)
    ind = np.argsort(w)[::-1] # 固有値の大きい順に並び替えた時のwのインデックス
    D = np.dot(X,v[:, ind[:k]])
    P = np.sum(w[ind[:k]])/np.sum(w) 
    return D, P


# In[13]:


from sklearn.datasets import load_wine
wine = load_wine()
X_wine=wine['data']
D, P=pca(X_wine, 2) # 2次元に縮約
print(P) # 累積寄与率

plt.figure(figsize=(7,5))
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.scatter(D[:,0], D[:,1],c=wine.target);


# In[8]:


#k-meansの実装
colorlist = ["y", "b", "g", "c", "m", "r", "k", "w"]
def squared_euclid(x, y):
    return np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y)
def kmeans(X, n_clusters, max_iter=20, rand_seed=0):
    count = 0
    np.random.seed()
    centers=X[np.random.choice(X.shape[0],n_clusters),:]
    for k in range(n_clusters):
        plt.scatter(centers[k:,0], centers[k:,1], c=colorlist[k])
    d = np.zeros((X.shape[0], n_clusters))
    clusters=np.zeros(X.shape[0])   
    for l in range(max_iter): 
        count += 1
        for i in range(X.shape[0]):
            for k in range(n_clusters):
                d[i,k] = squared_euclid(X[i], centers[k])
        clusters = np.argmin(d,axis=1)
        oldcenters = centers
        centers = np.array([[np.mean(X[clusters==k,0]),np.mean(X[clusters==k,1])] for k in range(len(centers))])
        diff = np.array([np.linalg.norm(oldcenters[i]-centers[i]) for i in range(k)]).max()
        for k in range(n_clusters):
            plt.scatter(centers[k:,0], centers[k:,1], c=colorlist[k])
        if diff < 0.1:
            print(count)
            break
    cost = np.sum((X - centers[np.argmin(d,axis=1),:])**2)
    plt.savefig('figure3.png')
    return clusters, centers, cost 


# In[12]:


clusters, centers, loss=kmeans(D, 3)
# print(loss)
plt.figure(figsize=(7,5))
    
plt.scatter(D[clusters==0,0],D[clusters==0,1],c='yellow', alpha=0.8)
plt.scatter(D[clusters==1,0],D[clusters==1,1],c='blue', alpha=0.8)
plt.scatter(D[clusters==2,0],D[clusters==2,1],c='green', alpha=0.8)
plt.scatter(centers[:,0], centers[:,1], c='red')
# plt.savefig('figure4.png')


# In[ ]:




