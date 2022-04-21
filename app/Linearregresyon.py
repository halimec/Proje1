#!/usr/bin/env python
# coding: utf-8

# In[1]:


# loading dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy


# In[2]:


data = pd.read_csv("C:/Users/halime/Desktop/label/KARATAY/KARATAYSON.csv",sep=",",thousands=".")


# In[3]:


data = data.replace(',','', regex=True)


# In[4]:


del data['Unnamed: 0']


# In[5]:


data["fiyat"] = pd.to_numeric(data["fiyat"])


# In[6]:


Y=data[['fiyat']]


# In[7]:


Y = pd.DataFrame(Y, columns = ['fiyat'])


# In[8]:


X= data[['il', 'ilce',  'OdaSayısı',
       'BinanınYASI', 'BulunduguKAT', 'BinadakiKatSAYISI', 'IsıtmaTıpı',
       'BanyoSAYISI'
       ]]


# In[9]:


X = pd.DataFrame(X, columns = ['il', 'ilce',  'OdaSayısı',
       'BinanınYASI', 'BulunduguKAT', 'BinadakiKatSAYISI', 'IsıtmaTıpı',
       'BanyoSAYISI'
       ])


# In[10]:


X = X.to_numpy()
Y = Y.to_numpy()

print(Y)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


reg = LinearRegression().fit(X, Y)


# In[13]:


reg.score(X, Y)


# In[14]:


reg.predict(np.array([[0,0,1,0,8,4,1,0]]))


# In[15]:


import pickle


# In[ ]:


with open('nisan2linearpkl.pkl', 'wb') as files:
    pickle.dump(reg, files)

