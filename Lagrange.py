#!/usr/bin/env python
# coding: utf-8

# # 双対上昇法及び拡張ラグランジュ乗数法の実装

# ## 必要なインポート

# In[3]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # 双対上昇法

# ## 必要な関数と変数

# In[13]:


xlist = []
ylist = []
zlist = []
x = 0
y = 0
z = 0
step = 4.7
oldf = 0
newf = 10000
def DualFunction(x,y,z):
    return 3*x**2+2*y**2+z*(x+y-1)


# In[14]:


count = 0
while abs(x+y-1) > 0.1 and count < 60:
    x = -z/6
    y = -z/4
    xlist.append(x)
    ylist.append(y)
    zlist.append(z)
    z = z + step*(x+y-1)
    newf = DualFunction(x,y,z)
    count += 1


# In[15]:


x = np.arange(-1.0, 1.0, 0.1) # x軸
y = np.arange(-1.0, 1.0, 0.1) # y軸
 
X, Y = np.meshgrid(x, y)
Z = 3*X**2 + 2*Y**2

plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(111)
plt.subplot(1,2,1)
cont = plt.contour(X, Y, Z, linestyles='dashed', levels=20)
plt.plot(xlist, ylist, "o-", color="k", label="value1 of data01")
plt.plot(x,1-x)
plt.xlim([-0.25,0.75])
plt.ylim([-0.25,0.75])
plt.subplot(1,2,2)
plt.plot(np.array(range(len(zlist))), zlist, "o-")
plt.show()


# In[7]:


np.array(zlist).mean()


# 考察
# 以上の検証によって分かったこと
# 元々、繰り返しの終わりをnewfとoldfの差の絶対値が0.01以下になればとしていたが、
# stepを1にした場合収束が終わらなかったこと

# ## 発散しないステップを実験的に調べる

# In[16]:


def DualFunction(x,y,z):
    return 3*x**2+2*y**2+z*(x+y-1)


# In[17]:


resultlist = []
steplist = []
countlist = []
for i in range(10,50):
    xlist = []
    ylist = []
    zlist = []
    x = 0
    y = 0
    z = 0
    step = i/10
    oldf = 0
    newf = 10000
    count = 0
    while abs(x+y-1) > 0.1 and count < 100:
        x = -z/6
        y = -z/4
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        z = z + step*(x+y-1)
        newf = DualFunction(x,y,z)
        count += 1
    steplist.append(step)
    countlist.append(count)


# In[19]:


plt.plot(steplist,countlist,"o-")


# # 拡張ラグランジュ

# ## 必要な関数と変数

# In[45]:


xlist = []
ylist = []
zlist = []
x = 0
y = 0
z = 0
c = 100.0
oldf = 0
newf = 10000
def DualFunction(x,y,z):
    return 3*x**2+2*y**2+z*(x+y-1)
def LagrangeFunction(x,y,z,c):
    return DualFunction(x,y,z) + (c*(x+y-1)**2)/2


# In[46]:


count = 0
while abs(x+y-1) > 0.1 and count < 10:
    x = (c-z)/(6+5*c/2)
    y = (c-z)/(4+5*c/3)
    xlist.append(x)
    ylist.append(y)
    z = z + c*(x+y-1)
    zlist.append(z)
    newf = DualFunction(x,y,z)
    count += 1
    print([x,y])


# In[47]:


x = np.arange(-1.0, 1.0, 0.1) # x軸
y = np.arange(-1.0, 1.0, 0.1) # y軸
 
X, Y = np.meshgrid(x, y)
Z = 3*X**2 + 2*Y**2

plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(111)
plt.subplot(1,2,1)
cont = plt.contour(X, Y, Z, linestyles='dashed', levels=20)
plt.plot(xlist, ylist, "o-", color="k", label="value1 of data01")
plt.plot(x,1-x)
plt.xlim([-0.25,0.75])
plt.ylim([-0.25,0.75])
plt.subplot(1,2,2)
plt.plot(np.array(range(len(zlist))), zlist, "o-")
plt.show()


# ## 発散しないステップを調べる

# In[42]:


def DualFunction(x,y,z):
    return 3*x**2+2*y**2+z*(x+y-1)
def LagrangeFunction(x,y,z,c):
    return DualFunction(x,y,z) + (c*(x+y-1)**2)/2


# In[43]:


resultlist = []
steplist = []
countlist = []
for i in range(10,1000):
    xlist = []
    ylist = []
    zlist = []
    x = 0
    y = 0
    z = 0
    c = i/10
    oldf = 0
    newf = 10000
    count = 0
    while abs(x+y-1) > 0.1 and count < 100:
        x = (c-z)/(6+5*c/2)
        y = (c-z)/(4+5*c/3)
        xlist.append(x)
        ylist.append(y)
        zlist.append(z)
        z = z + c*(x+y-1)
        newf = DualFunction(x,y,z)
        count += 1
    steplist.append(c)
    countlist.append(count)


# In[44]:


plt.plot(steplist,countlist,"o-")


# In[ ]:




