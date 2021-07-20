#!/usr/bin/env python
# coding: utf-8

# # パターン情報学プログラミング課題１
# ### githubにまとめるため知能システム論のフォルダに移行

# In[2]:


# モジュールのインポート
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 課題１−１（k近傍法の実装）

# In[3]:


from sklearn.datasets import load_iris
iris = load_iris() # データセットのロード


# In[4]:


print(len(iris['data'])) # データの総数
print(iris['feature_names']) # 特徴量名
print(iris['data'][0:5]) # データ（先頭5行を表示）
print(iris['target_names']) # ラベル名
print(iris['target'][0:5], iris['target'][50:55], iris['target'][100:105]) # ラベル


# In[5]:


data = iris['data']
target = iris['target']
data = np.insert(data, 4, target, axis=1)
idrate_list = []
for k in range(30):
    id_rate = 0
    for i in range(len(data)):
        test_data = data[i]
        train_data = np.delete(data,i,0)
        labeling_data = np.array(sorted(train_data, key=lambda x:np.linalg.norm(test_data[0:4]-x[0:4]))[:k+1])
        labelcountlist = [np.sum(labeling_data[:,4] == 0),np.sum(labeling_data[:,4] == 1),np.sum(labeling_data[:,4] == 2)]
        label = np.argmax(labelcountlist)
        if label == test_data[4]:
            id_rate += 1
    idrate_list.append(id_rate/len(data))


# In[6]:


plt.xlabel('k')
plt.ylabel('identification-rate')
plt.plot(np.arange(1,31),idrate_list)


# In[7]:


idrate_list
#19,20,21が最適


# ## 課題１−２（k-meansの実装）

# In[8]:


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


# In[9]:


iris = load_iris() 
X_iris=iris['data'][:,[2,3]]
clusters, centers, loss=kmeans(X_iris, 5)
print(loss)
plt.figure(figsize=(7,5))
plt.xlabel(iris["feature_names"][2])
plt.ylabel(iris["feature_names"][3])
    
plt.scatter(X_iris[clusters==0,0],X_iris[clusters==0,1],c='yellow', alpha=0.2)
plt.scatter(X_iris[clusters==1,0],X_iris[clusters==1,1],c='blue', alpha=0.2)
plt.scatter(X_iris[clusters==2,0],X_iris[clusters==2,1],c='green', alpha=0.2)
plt.scatter(X_iris[clusters==3,0],X_iris[clusters==3,1],c='black', alpha=0.2)
plt.scatter(X_iris[clusters==4,0],X_iris[clusters==4,1],c='orange', alpha=0.2)
plt.scatter(centers[:,0], centers[:,1], c='red')
plt.savefig('figure4.png')


# In[10]:


centers[:,0]


# ## 課題１−３（線形重回帰の実装）

# In[96]:


#データの前処理
from scipy import stats
filename = "auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin', 'name']
df = pd.read_csv(filename, delim_whitespace=True, names=column_names)
df = df.drop(df['horsepower'][df['horsepower'] == '?'].index)
df = df.reset_index(drop=True)
df['horsepower'] = [float(df['horsepower'][i]) for i in range(len(df['horsepower']))]


# ### 指定された二変数での実装

# In[145]:


# ex_val = df[['weight','horsepower']].iloc[:,:].apply(stats.zscore, axis=0)
# ob_val = df[['mpg']].iloc[:].apply(stats.zscore, axis=0)
ex_val = df[['weight','horsepower']]
ob_val = df['mpg']
ex_val = np.array(ex_val)
ex_val = np.insert(ex_val,0,1,axis = 1)
ob_val = np.array(ob_val)
#回帰係数の算出
reg_cof = np.dot(np.dot(np.linalg.inv(np.dot(ex_val.T,ex_val)),ex_val.T),ob_val)
#回帰式
linear = np.zeros(len(ob_val))
for k in range(3):
    linear += reg_cof[k]*ex_val[:,k]
# linear = reg_cof[1]*ex_val[:,1] + reg_cof[2]*ex_val[:,2] + reg_cof[0]*ex_val[:,0]
#決定係数score算出
child = np.dot(linear-ex_val[:,0]*np.mean(ob_val),linear-ex_val[:,0]*np.mean(ob_val))
mother = np.dot(ob_val-ex_val[:,0]*np.mean(ob_val),ob_val-ex_val[:,0]*np.mean(ob_val))
score = child/mother
print("回帰係数："+str(reg_cof[1])+"、"+str(reg_cof[2]))
print("切片："+str(reg_cof[0]))
print("決定係数："+str(score))


# ### 他の変数も含めた四変数での実装

# In[146]:


# ex_val = df[['weight','horsepower']].iloc[:,:].apply(stats.zscore, axis=0)
# ob_val = df[['mpg']].iloc[:].apply(stats.zscore, axis=0)
ex_val = df[['weight','horsepower','displacement','acceleration']]
ob_val = df['mpg']
ex_val = np.array(ex_val)
ex_val = np.insert(ex_val,0,1,axis = 1)
ob_val = np.array(ob_val)
#回帰係数の算出
reg_cof = np.dot(np.dot(np.linalg.inv(np.dot(ex_val.T,ex_val)),ex_val.T),ob_val)
#回帰式
linear = np.zeros(len(ob_val))
for n in range(ex_val.shape[1]):
    linear += reg_cof[n]*ex_val[:,n]
#決定係数score算出
child = np.dot(linear-ex_val[:,0]*np.mean(ob_val),linear-ex_val[:,0]*np.mean(ob_val))
mother = np.dot(ob_val-ex_val[:,0]*np.mean(ob_val),ob_val-ex_val[:,0]*np.mean(ob_val))
score = child/mother
#回帰係数のプリント
reg_cofstr = '回帰係数：'
for n in range(1,ex_val.shape[1]):
    reg_cofstr += (str(reg_cof[n])+' ')
print(reg_cofstr)
print("切片："+str(reg_cof[0]))
print("決定係数："+str(score))


# ### 正規化した時の回帰係数導出

# In[157]:


# ex_val = df[['weight','horsepower']].iloc[:,:].apply(stats.zscore, axis=0)
# ob_val = df[['mpg']].iloc[:].apply(stats.zscore, axis=0)
ex_val = df[['weight','horsepower','displacement','acceleration']]
ob_val = df['mpg']
ex_val = np.array(ex_val)
ex_val = np.insert(ex_val,0,1,axis = 1)
ob_val = np.array(ob_val)
for i in range(1,ex_val.shape[1]):
    ex_val[:,i] = (ex_val[:,i]-np.mean(ex_val[:,i]))/np.std(ex_val[:,i])
ob_val = (ob_val-np.mean(ob_val))/np.std(ob_val)
# ex_val = scipy.stats.zscore(ex_val, ddof=1)
# ob_val = scipy.stats.zscore(ob_val, ddof=1)
#回帰係数の算出
reg_cof = np.dot(np.dot(np.linalg.inv(np.dot(ex_val.T,ex_val)),ex_val.T),ob_val)
#回帰式
linear = np.zeros(len(ob_val))
for n in range(ex_val.shape[1]):
    linear += reg_cof[n]*ex_val[:,n]
#決定係数score算出
child = np.dot(linear-ex_val[:,0]*np.mean(ob_val),linear-ex_val[:,0]*np.mean(ob_val))
mother = np.dot(ob_val-ex_val[:,0]*np.mean(ob_val),ob_val-ex_val[:,0]*np.mean(ob_val))
score = child/mother
#回帰係数のプリント
reg_cofstr = '回帰係数：'
for n in range(1,ex_val.shape[1]):
    reg_cofstr += (str(reg_cof[n])+' ')
print(reg_cofstr)
print("切片："+str(reg_cof[0]))
print("決定係数："+str(score))


# ### 可視化（二変数の時）

# In[88]:


from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt
import seaborn as sns

fig=plt.figure()
ax=Axes3D(fig)
# x1 = df['weight']
# x2 = df['horsepower']
# y = df['mpg']
x1 = ex_val[:,1]
x2 = ex_val[:,2]
y = ob_val

ax.scatter3D(x1, x2, y)
# X1, X2 = np.meshgrid(x1, x2)
# Y = reg_cof[0]*X1 + reg_cof[1]*X2
x = np.arange(1000, 5000, 500)
y = np.arange(25, 250, 25)
X, Y = np.meshgrid(x, y)
Z = reg_cof[1]*X + reg_cof[2]*Y + reg_cof[0]
fig = plt.figure()
ax.set_xlabel("weight")
ax.set_ylabel("horsepower")
ax.set_zlabel("mpg")
ax.set_xlim(1000, 5000)
ax.set_ylim(0, 250)
ax.set_zlim(5, 45)
ax.plot_wireframe(X, Y, Z, color='g')
ax.view_init(elev = 10, azim=-55)
plt.show()


# ### 既存のライブラリを用いた確認

# In[147]:


from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import preprocessing
X = df[['weight','horsepower','displacement','acceleration']]
Y = df['mpg']
model_lr = LinearRegression()
model_lr.fit(X, Y)
print("回帰係数= ",model_lr.coef_)
print("切片= ",model_lr.intercept_)
print("決定係数= ",model_lr.score(X, Y))


# In[90]:


from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt
import seaborn as sns

fig=plt.figure()
ax=Axes3D(fig)
x1 = df['weight']
x2 = df['horsepower']
y = df['mpg']

ax.scatter3D(x1, x2, y)
# X1, X2 = np.meshgrid(x1, x2)
# Y = reg_cof[0]*X1 + reg_cof[1]*X2
x = np.arange(1000, 5000, 500)
y = np.arange(25, 250, 25)
X, Y = np.meshgrid(x, y)
Z = model_lr.coef_[0]*X + model_lr.coef_[1]*Y + 45.64
fig = plt.figure()
ax.plot_wireframe(X,Y,Z,color = 'g')
ax.set_xlabel("weight")
ax.set_ylabel("horsepower")
ax.set_zlabel("mpg")
ax.set_xlim(1000, 5000)
ax.set_ylim(0, 250)
ax.set_zlim(5, 45)
# ax.plot_wireframe(X1, X2, Y, color='g')
ax.view_init(elev=10, azim=-55)
plt.show()


# In[ ]:




