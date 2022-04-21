#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy


# In[2]:


data = pd.read_csv("C:/Users/halime/Desktop/label/KARATAY/KARATAYSON.csv",sep=",",thousands=".")


# In[3]:


data.columns


# In[4]:


data.dtypes


# In[5]:


data = data.replace(',','', regex=True)


# In[6]:


data["fiyat"] = pd.to_numeric(data["fiyat"])


# In[7]:


data.dtypes


# In[8]:


Y=data[['fiyat']]


# In[9]:


Y = pd.DataFrame(Y, columns = ['fiyat'])


# In[10]:


X= data[['il', 'ilce',  'OdaSayısı',
       'BinanınYASI', 'BulunduguKAT', 'BinadakiKatSAYISI', 'IsıtmaTıpı',
       'BanyoSAYISI'
       ]]


# In[11]:


X = pd.DataFrame(X, columns = ['il', 'ilce',  'OdaSayısı',
       'BinanınYASI', 'BulunduguKAT', 'BinadakiKatSAYISI', 'IsıtmaTıpı',
       'BanyoSAYISI'
       ])


# In[12]:


X = X.to_numpy()
Y = Y.to_numpy()

print(Y)


# In[13]:


data.dtypes


# In[14]:


del data['Unnamed: 0']


# In[15]:


data.dtypes


# In[16]:


from sklearn.naive_bayes import GaussianNB


# In[17]:


clf1 = GaussianNB()


# In[18]:


clf1.fit(X, Y)


# In[19]:


print(clf1.predict([[0,0,1,0,8,4,1,0]]))


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109) # 70% training and 30% test


# In[22]:


clf1.fit(X_train, y_train)


# In[23]:


y_pred = clf1.predict(X_test)


# In[24]:


from sklearn import metrics


# In[25]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


import pickle


# In[ ]:


with open('nisan2gaussianpkl.pkl', 'wb') as files:
    pickle.dump(clf1, files)


# In[ ]:




