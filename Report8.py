#!/usr/bin/env python
# coding: utf-8

# # 宿題１（交差確認法）

# In[1]:


import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt


# ## コストの算出

# In[2]:


# np.random.seed(0)
repeat = 10
#これ使いませんでした
def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

losslist = np.array([np.array([i,j,np.zeros(1)[0]]) for i in range(1,21) for j in range(1,101)])
for n in range(repeat):
    sample_size = 50
    xmin, xmax = -3, 3
    split = 5
    x = 6*(np.arange(sample_size)/(sample_size-1)-1/2)
    x = np.random.permutation(x)
    y = np.sin(np.pi*x)/(np.pi*x)+0.1*x+0.1*randn(sample_size)
    X = np.linspace(start=xmin, stop=xmax, num=5000)
    predictionlist = np.zeros(len(X)).reshape(len(X),-1)
    for h in range(1,21):
        for l in range(1,101):
            loss = 0
            for i in range(split):
                x_test = x[int(i*len(x)/split):int((i+1)*len(x)/split)]
                y_test = y[int(i*len(y)/split):int((i+1)*len(y)/split)]
                x_pre = np.delete(x, slice(int(i*len(x)/split), int((i+1)*len(x)/split)))
                y_pre = np.delete(y, slice(int(i*len(y)/split), int((i+1)*len(y)/split)))
                # calculate design matrix
                k = calc_design_matrix(x_pre, x_pre, h/10)
                # solve the least square problem
                theta = np.linalg.solve(k.T.dot(k) + l/1000 * np.identity(len(k)),k.T.dot(y_pre[:, None]))
                # create data to visualize the prediction
#                 K = calc_design_matrix(x_pre, X, h/10)
                K_test = calc_design_matrix(x_pre, x_test, h/10)
#                 prediction = np.dot(K,theta)
                predict_test = np.dot(K_test,theta)
#                 predictionlist[:,0] += prediction[:,0]
                loss += np.dot(predict_test[:,0]-y_test,predict_test[:,0]-y_test)/len(y_test)
            losslist[(100*(h-1)+(l-1)),2] += loss
losslist[:,2] = losslist[:,2]/repeat


# ## 各h,lに対するスコアの可視化（三次元プロット）

# In[3]:


from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt
import seaborn as sns

fig=plt.figure()
ax=Axes3D(fig)
x = losslist[:,0]/10
y = losslist[:,1]/1000
z = losslist[:,2]

ax.scatter3D(x, y, z)
ax.set_xlabel("h")
ax.set_ylabel("l")
ax.set_zlabel("loss")
ax.view_init(elev = 0, azim=90)
plt.show()


# ## 最も優れているhとlの値の出力

# In[4]:


losslist[np.argmin(losslist[:,2])]


# ## ベストスコアを出したものの可視化

# In[11]:


# np.random.seed(0)
h = 9
l = 6
xmin, xmax = -3, 3
split = 5
x = 6*(np.arange(sample_size)/(sample_size-1)-1/2)
x = np.random.permutation(x)
y = np.sin(np.pi*x)/(np.pi*x)+0.1*x+0.1*randn(sample_size)
X = np.linspace(start=xmin, stop=xmax, num=5000)
predictionlist = np.zeros(len(X)).reshape(len(X),-1)
loss = 0
for i in range(split):
    x_test = x[int(i*len(x)/split):int((i+1)*len(x)/split)]
    y_test = y[int(i*len(y)/split):int((i+1)*len(y)/split)]
    x_pre = np.delete(x, slice(int(i*len(x)/split), int((i+1)*len(x)/split)))
    y_pre = np.delete(y, slice(int(i*len(y)/split), int((i+1)*len(y)/split)))
    # calculate design matrix
    k = calc_design_matrix(x_pre, x_pre, h/10)
    # solve the least square problem
    theta = np.linalg.solve(k.T.dot(k) + l/1000 * np.identity(len(k)),k.T.dot(y_pre[:, None]))
    # create data to visualize the prediction
    K = calc_design_matrix(x_pre, X, h/10)
    K_test = calc_design_matrix(x_pre, x_test, h/10)
    prediction = np.dot(K,theta)
    predict_test = np.dot(K_test,theta)
    predictionlist[:,0] += prediction[:,0]
    loss += np.dot(predict_test[:,0]-y_test,predict_test[:,0]-y_test)/len(y_test)
print(loss)
prediction[:,0] = predictionlist[:,0]/split
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)


# ## 値が大きくずれている場合

# In[6]:


np.random.seed(0)
h = 100
l = 0.1
xmin, xmax = -3, 3
split = 5
x = 6*(np.arange(sample_size)/(sample_size-1)-1/2)
x = np.random.permutation(x)
y = np.sin(np.pi*x)/(np.pi*x)+0.1*x+0.1*randn(sample_size)
X = np.linspace(start=xmin, stop=xmax, num=5000)
predictionlist = np.zeros(len(X)).reshape(len(X),-1)
loss = 0
for i in range(split):
    x_test = x[int(i*len(x)/split):int((i+1)*len(x)/split)]
    y_test = y[int(i*len(y)/split):int((i+1)*len(y)/split)]
    x_pre = np.delete(x, slice(int(i*len(x)/split), int((i+1)*len(x)/split)))
    y_pre = np.delete(y, slice(int(i*len(y)/split), int((i+1)*len(y)/split)))
    # calculate design matrix
    k = calc_design_matrix(x_pre, x_pre, h/10)
    # solve the least square problem
    theta = np.linalg.solve(k.T.dot(k) + l/100 * np.identity(len(k)),k.T.dot(y_pre[:, None]))
    # create data to visualize the prediction
    K = calc_design_matrix(x_pre, X, h/10)
    K_test = calc_design_matrix(x_pre, x_test, h/10)
    prediction = np.dot(K,theta)
    predict_test = np.dot(K_test,theta)
    predictionlist[:,0] += prediction[:,0]
    loss += np.dot(predict_test[:,0]-y_test,predict_test[:,0]-y_test)
print(loss)
prediction[:,0] = predictionlist[:,0]/split
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)


# ## 宿題２（SVMの劣勾配アルゴリズム実装）

# In[7]:


import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(sample_size):
    a = np.linspace(0, 4 * np.pi, num=sample_size // 2)
    x = np.concatenate(
        [np.stack([a * np.cos(a), a * np.sin(a)], axis=1),
         np.stack([(a + np.pi) * np.cos(a), (a + np.pi) * np.sin(a)], axis=1)])
    x += np.random.random(size=x.shape)
    y = np.concatenate([np.ones(sample_size // 2), -np.ones(sample_size // 2)])
    return x, y


def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


def optimize_param(design_mat, y, regularizer, lr): 
    theta = np.dot(np.linalg.inv(design_mat),y)
    for i in range(5):
        array1 = np.dot(y.T,design_mat)
        array1 = np.where(array1<0,0,array1)
        theta = theta - lr*(regularizer*array1+2*np.dot(design_mat,theta))
    return theta
#lr: learning rate
# implement here


def visualize(theta, x, y, grid_size=100, x_min=-16, x_max=16):
    grid = np.linspace(x_min, x_max, grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    design_mat = build_design_mat(x, mesh_grid, bandwidth=1.)
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.contourf(X, Y, np.reshape(np.sign(design_mat.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$O$', c='blue')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='x', c='red')
    plt.savefig('ML1-homework2.png')

x, y = generate_data(sample_size=200)
design_mat = build_design_mat(x, x, bandwidth=1.)
theta = optimize_param(design_mat, y, regularizer=1, lr=0.0001)
visualize(theta,x,y)


# In[ ]:




