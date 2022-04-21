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


data = data.replace(',','', regex=True)


# In[4]:


data["fiyat"] = pd.to_numeric(data["fiyat"])


# In[5]:


Y=data[['fiyat']]


# In[6]:


del data['Unnamed: 0']


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


from sklearn import svm


# In[12]:


clf = svm.SVC()


# In[13]:


clf.fit(X, Y)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109) # 70% training and 30% test


# In[15]:


clf.fit(X_train, y_train)


# In[16]:


y_pred = clf.predict(X_test)


# In[17]:


from sklearn import metrics


# In[18]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


import pickle


# In[ ]:


with open('nisan3svmmodelpkl.pkl', 'wb') as files:
    pickle.dump(clf, files)


# In[ ]:


with open('nisan3svmmodelpkl.pkl' , 'rb') as f:
    clf = pickle.load(f)


clf.predict([[0,0,1,0,8,4,1,0]]) 

