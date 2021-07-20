#!/usr/bin/env python
# coding: utf-8

# # バックトラック探索を用いた勾配法実装

# ## 必要なインポート

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import random


# ## 関数定義

# In[8]:


#イプシロンの決定関数
def epsilon_decide(x1,x2,beta):
    epsilon = 1
    while True:
        if epsilon <= (200*(x1**2)+2*(x2**2))/(4000*(x1**2)+4*(x2**2)):
            return epsilon
        else:
            epsilon = epsilon*beta
#変数の更新関数
def variable_revise(x1,x2,epsilon):
    global oldvariable1
    global oldvariable2
    oldvariable1 = x1
    oldvariable2 = x2
    x1 = x1 - epsilon*20*x1
    x2 = x2 - epsilon*2*x2
    global newvariable1
    global newvariable2
    global x1list
    global x2list
    newvariable1 = x1
    newvariable2 = x2
    x1list.append(oldvariable1)
    x2list.append(oldvariable2)

#xをこれ以上更新するかジャッジする関数
#規準としてユークリッド距離が10**(-2)になったら更新をストップするものとする
#更新するか否かをBool型で返すものとする
def revise_judge(newx1,newx2,oldx1,oldx2):
    if abs(newx1-oldx1)+abs(newx2-oldx2) < 10**(-2):
        x1list.append(newvariable1)
        x2list.append(newvariable2)
        return True
    else:
        return False


# ## 変数宣言

# In[9]:


#更新後の変数
newvariable1 = 1.0
newvariable2 = 5.0
#一つ前の変数
oldvariable1 = 0
oldvariable2 = 0
#採用されたイプシロン
usedepsilon = 0
#プロット用のリスト
x1list = []
x2list = []


# ## 探索かいし

# In[10]:


while(revise_judge(newvariable1,newvariable2,oldvariable1,oldvariable2) == False):
    usedepsilon = epsilon_decide(newvariable1,newvariable2,0.8)
    variable_revise(newvariable1,newvariable2,usedepsilon)
    revise_judge(newvariable1,newvariable2,oldvariable1,oldvariable2)


# ## 探索点の図示

# In[15]:


x = np.arange(-7.5, 7.5, 0.1) # x軸
y = np.arange(-7.5, 7.5, 0.1) # y軸
 
X, Y = np.meshgrid(x, y)
Z = 10*X**2 + Y**2

plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(111)
plt.subplot(1, 2, 1)
plt.plot(x1list, x2list, "o-", color="k", label="value1 of data01")
cont = plt.contour(X, Y, Z, linestyles='dashed', levels=20)
plt.show()


# In[ ]:




