#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()


# In[2]:


X=data.iloc[:,:-1].values
X


# In[3]:


y=data.iloc[:,-1].values
y


# In[5]:


data.isnull().sum()


# In[6]:


data.duplicated()


# In[7]:


data.describe()


# In[8]:


data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()


# In[9]:


scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)


# In[11]:


X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[12]:


X_train


# In[14]:


X_test


# In[15]:


print("Lenght of X_test ",len(X_test))


# In[ ]:




