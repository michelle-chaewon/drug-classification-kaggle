#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd


# In[3]:


os.chdir(r"C:\Users\bakch\Downloads\archive (2)")


# In[4]:


df = pd.read_csv('drug200.csv')

df.columns


# BP: Blood Pressure Levels  
# Na_to_K: Sodium to potassium Ratio in Blood  

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


sns.countplot(df.Drug)


# In[103]:


bins=[0,10,20,30,40,50,60,70,80,90,100]

plt.hist(df.Age, bins, histtype='bar', rwidth=0.9)

plt.xlabel('Age')
plt.ylabel('Count')

plt.grid(True)
plt.show()


# In[18]:


sns.countplot(x='Drug', hue='BP', data=df)


# In[104]:


sns.countplot(x='Drug', hue='Sex', data=df)


# In[105]:


sns.countplot(x='Drug', hue='Cholesterol', data=df)


# Logistic Regression

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Drug', axis=1), df['Drug'], test_size=0.30, random_state=1)


# In[12]:


# Get list of categorical variables
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
object_cols_train = object_cols[0:3]

print("Categorical variables:")
print(object_cols_train)


# In[17]:


#cardinality
low_card_cols = [col for col in object_cols_train if X_train[col].nunique() < 10]


# In[13]:


#ordinal: BP, Cholesterol
#non-ordinal: Sex, Drug


# In[14]:


from sklearn.preprocessing import OneHotEncoder


# In[18]:


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_card_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_card_cols]))


# In[20]:


# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index


# In[27]:


# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols_train, axis=1)
num_X_test = X_test.drop(object_cols_train, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)


# In[87]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs',multi_class='multinomial', class_weight='balanced', max_iter=2200)


# In[88]:


logmodel.fit(OH_X_train,y_train)
predictions = logmodel.predict(OH_X_test)


# In[89]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




