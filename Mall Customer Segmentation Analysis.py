#!/usr/bin/env python
# coding: utf-8

# # Mall Customer Segmentation Analysis
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("C:/Users/rockz/OneDrive/Documents/Mall Customers.csv")
data


# In[3]:


data.info()


# In[4]:


data.nunique()


# In[5]:


data.columns


# In[6]:


data.describe()


# # Visualization

# In[7]:


plt.figure(figsize=(10,10))
plt.subplot(4,2,1)
sns.histplot(data=data['Age'], x=data['Age'],kde=True)
plt.subplot(4,2,2)
sns.histplot(data=data['Annual Income (k$)'],x=data['Annual Income (k$)'],kde=True)
plt.subplot(4,2,5)
sns.histplot(data=data['Spending Score (1-100)'],x=data['Spending Score (1-100)'],kde=True)


# In[8]:


plt.figure(1, figsize = (15, 5))
sns.countplot(y = 'Gender',data = data)
plt.show()


# In[9]:


plt.figure(1)
for gender in ['Male','Female']:
    plt.scatter(x='Annual Income (k$)',y='Spending Score (1-100)',data=data[data['Gender']==gender],s=100,alpha=0.5,label=gender)
plt.xlabel('Annual Income (k$)'),plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score w.r.t Gender')
plt.legend()
plt.show()


# In[10]:


plt.figure(1)
for gender in ['Male','Female']:
    plt.scatter(x='Age',y='Annual Income (k$)',data=data[data['Gender']==gender],s=100,alpha=0.5,label=gender)
plt.xlabel('Age'),plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income w.r.t Gender')
plt.legend()
plt.show()


# In[11]:


plt.figure(1,figsize=(10,6))
n=0 
for cols in ['Age','Annual Income (k$)','Spending Score (1-100)']:
    n += 1 
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.violinplot(x=cols,y='Gender',data=data,palette='vlag')
    sns.swarmplot(x=cols,y='Gender',data=data)
    plt.ylabel('Gender' if n==1 else'')
    plt.title('Boxplots & Swarmplots'if n==2 else'')
plt.show()


# In[12]:


plt.figure(1)
for gender in ['Male','Female']:
    plt.scatter(x='Age',y='Spending Score (1-100)',data=data[data['Gender']==gender],s=100,alpha=0.5,label=gender)
plt.xlabel('Age'),plt.ylabel('Spending Score (1-100)') 
plt.title('Age vs Spending Score w.r.t Gender')
plt.legend()
plt.show()


# In[13]:


x=data[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']].values
x[0:5]


# In[14]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['Male','Female'])
x[:,0] = le.transform(x[:,0])


# In[15]:


x[0:5]


# # Modelling
# ## K-Means

# In[16]:


from sklearn.cluster import KMeans


# In[17]:


x1=data[['Age','Spending Score (1-100)']].iloc[:,:].values
inertia=[]
for n in range(1,11):
    algorithm=(KMeans(n_clusters=n,init='k-means++',n_init=10,max_iter=300,tol=0.0001,random_state=111))
    algorithm.fit(x1)
    inertia.append(algorithm.inertia_)


# In[18]:


plt.figure(1)
plt.plot(np.arange(1,11),inertia,'o')
plt.plot(np.arange(1,11),inertia,'-',alpha=0.5)
plt.xlabel('Number of Clusters'),plt.ylabel('Inertia')
plt.show()


# In[20]:


algorithm=(KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,tol=0.0001,random_state=111))


# In[24]:


pred=algorithm.fit_predict(x1)
pred


# In[29]:


data['cluster']=pred
data


# In[30]:


m=algorithm.cluster_centers_
m


# In[33]:


sns.scatterplot(x=data['Age'],y=data['Annual Income (k$)'],hue=data['cluster'])
plt.scatter(x=m[:,0],y=m[:,1],marker="*",s=100,color="red")


# In[ ]:




