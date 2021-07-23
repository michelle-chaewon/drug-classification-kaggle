#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd


# In[6]:


os.chdir(r"C:\Users\bakch\Downloads\archive (2)")


# In[7]:


df = pd.read_csv('drug200.csv')

df.columns


# BP: Blood Pressure Levels  
# Na_to_K: Sodium to potassium Ratio in Blood  

# In[8]:


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


# In[124]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# Logistic Regression

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Drug', axis=1), df['Drug'], test_size=0.30, random_state=1)


# In[12]:


# Get list of categorical variables
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
object_cols_train = object_cols[0:3]

print("Categorical variables:")
print(object_cols_train)


# In[13]:


#cardinality
low_card_cols = [col for col in object_cols_train if X_train[col].nunique() < 10]


# In[14]:


#ordinal: BP, Cholesterol
#non-ordinal: Sex, Drug


# In[15]:


from sklearn.preprocessing import OneHotEncoder


# In[16]:


# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_card_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_card_cols]))


# In[17]:


# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index


# In[18]:


# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols_train, axis=1)
num_X_test = X_test.drop(object_cols_train, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)


# In[118]:


from sklearn.linear_model import LogisticRegression


# In[338]:


pipeline_lr = Pipeline([('scalar1', StandardScaler()),
                      ('pca1', PCA(n_components=0.95)),
                      ('lr_classifier', LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial', class_weight='balanced', max_iter=2200))])

pipeline_dt = Pipeline([('scalar2', StandardScaler()),
                      ('pca2', PCA(n_components=0.95)),
                      ('dt_classifier', DecisionTreeClassifier())])

pipeline_randomforest = Pipeline([('scalar3', StandardScaler()),
                      ('pca3', PCA(n_components=0.95)),
                      ('rf_classifier', RandomForestClassifier())])


# In[ ]:


pipelines = [pipeline_lr, pipeline_dt, pipeline_randomforest]


# In[339]:


best_accuracy = 0.0
best_classifier = 0
best_pipeline = ""


# In[347]:


pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest'}

for pipe in pipelines:
    pipe.fit(OH_X_train, y_train)


# In[349]:


from sklearn.model_selection import KFold

cv = KFold(n_splits=10, random_state=1, shuffle=True)


# In[350]:


for i, model in enumerate(pipelines):
    scores = cross_val_score(model, OH_X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1).mean()
    for score in np.nditer(scores):    
        if score > best_accuracy:
            best_accuracy = score
            best_pipeline = pipe_dict[i]
            best_classifier = i

print('Classifier with best accuracy: {}'.format(pipe_dict[best_classifier]))
print('Accuracy: {}'.format(best_accuracy))


# In[ ]:




