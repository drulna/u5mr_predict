#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pal = sns.color_palette()


# In[2]:


original_data = pd.read_csv('data/updtedDataset.txt', sep='\t', encoding='utf-8', low_memory=False)


# In[3]:


print('Total records before removing NIU:', len(original_data))


# In[4]:


original_data = original_data[original_data["Childs age at death in month (including imputed)"] != 'NIU (not in universe)']


# In[5]:


print('Total records after removing NIU:', len(original_data))


# # Separate class label

# In[6]:


y = original_data["Childs age at death in month (including imputed)"]


# # Drop identifiers

# In[7]:


X = original_data.drop(['SAMPLE','COUNTRY', 'Year of Sample', 'Unique cross-sample respondent identifier','Unique cross-sample household identifier', 'Key to link DHS clusters to context data (string)', 'Unique sample-case PSU identifier', 'CASEID', 'Unique cross-sample sampling strata', 'Birth history index number', 'Sample weight for persons', 'All woman factor for total population', 'Weight for domestic violence module', 'Usual resident or visitor', 'Weight for age percentile (CDC standards)', 'Height for Age percentile (CDC standards)', 'Weight for height percentile (CDC standards)'], axis=1)
X = X.drop(["Childs age at death in month (including imputed)"], axis=1)


# In[8]:


column_names = list(X.columns.values)
X.columns = column_names


# In[9]:


X = X.replace('NIU (not in universe)', 'NIU')
X =  (X.replace(r'\s+', np.nan, regex=True))


# In[10]:


X[['Doctor gave delivery care','Relative gave delivery care', 'Traditional birth attendant gave delivery care', 'Trained traditional birth attendant gave delivery care', 'Nurse/midwife gave delivery care', 'Auxiliary midwife gave delivery care']] = X[['Doctor gave delivery care','Relative gave delivery care', 'Traditional birth attendant gave delivery care', 'Trained traditional birth attendant gave delivery care', 'Nurse/midwife gave delivery care', 'Auxiliary midwife gave delivery care']].replace(['Missing', 'nan'], ['NIU', 'NIU'])


# In[11]:


X[['Nurse/midwife gave prenatal care', 'Doctor gave prenatal care', 'Traditional birth attendant gave prenatal care']] = X[['Nurse/midwife gave prenatal care', 'Doctor gave prenatal care', 'Traditional birth attendant gave prenatal care']].replace(['Missing', 'nan'], ['NIU', 'NIU'])


# In[12]:


X[['Source of fever/cough treatment: Public hospital', 'Source of fever/cough treatment: Public health center', 'Source of fever/cough treatment: Traditional healer/practitioner', 'Source of fever/cough treatment: Private hospital/clinic']] = X[['Source of fever/cough treatment: Public hospital', 'Source of fever/cough treatment: Public health center', 'Source of fever/cough treatment: Traditional healer/practitioner', 'Source of fever/cough treatment: Private hospital/clinic']].replace(['Missing', 'nan'], ['NIU', 'NIU'])


# In[ ]:





# In[13]:


X = X.fillna(axis='index', method='ffill')


# In[ ]:





# In[14]:


X['Skilled prenatal given by skilled provider'] = X[['Nurse/midwife gave prenatal care', 'Doctor gave prenatal care', 'Traditional birth attendant gave prenatal care']].fillna('').max(axis=1)


# In[15]:


X['Skilled provider gave Delivery care'] = X[['Doctor gave delivery care','Relative gave delivery care', 'Traditional birth attendant gave delivery care', 'Trained traditional birth attendant gave delivery care', 'Nurse/midwife gave delivery care', 'Auxiliary midwife gave delivery care']].fillna('').max(axis=1)


# In[16]:


X['Child received care for fever/cough'] = X[['Source of fever/cough treatment: Public hospital', 'Source of fever/cough treatment: Public health center', 'Source of fever/cough treatment: Traditional healer/practitioner', 'Source of fever/cough treatment: Private hospital/clinic']].fillna('').max(axis=1)


# In[ ]:





# In[17]:


X = X.drop(['Nurse/midwife gave prenatal care', 'Doctor gave prenatal care', 'Traditional birth attendant gave prenatal care'], axis=1)


# In[18]:


X = X.drop(['Doctor gave delivery care','Relative gave delivery care', 'Traditional birth attendant gave delivery care', 'Trained traditional birth attendant gave delivery care', 'Nurse/midwife gave delivery care', 'Auxiliary midwife gave delivery care'], axis=1)


# In[19]:


X = X.drop(['Source of fever/cough treatment: Public hospital', 'Source of fever/cough treatment: Public health center', 'Source of fever/cough treatment: Traditional healer/practitioner', 'Source of fever/cough treatment: Private hospital/clinic'], axis = 1)


# In[20]:


X.tail()


# In[21]:


column_names = list(X.columns.values)


# In[22]:


X = X.astype(str)


# In[23]:


X.to_csv('data/clean_data.tsv', sep = '\t', index=False, encoding='utf-8')
X = np.array(X)
print('Total Records: ', X.shape[0])
print('Total Variables/Attributes: ', X.shape[1])

y = np.array(y)
print(y.shape)


# In[24]:


np.save('data/labels.npy', y)


# # Plot Class Distribution

# In[25]:


fig = plt.figure(figsize=(25,10))
ax = original_data.groupby("Childs age at death in month (including imputed)")["Childs age at death in month (including imputed)"].count().plot.bar()
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    
plt.ylabel('count')
plt.show()


# ### So much labels are set as (NIU)

# ##### Let's look at log scaled

# In[26]:


fig = plt.figure(figsize=(25,10))
ax = original_data.groupby("Childs age at death in month (including imputed)")["Childs age at death in month (including imputed)"].count().plot.bar()
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    
ax.set_yscale('log')
plt.ylabel('count (log-scaled)')
plt.show()


# # Plot Feature Importance

# In[27]:


from sklearn.preprocessing import LabelEncoder


# In[28]:


X = np.nan_to_num(X)
for i in range (X.shape[1]):
    print(np.unique(X[:,i]))
    le = LabelEncoder()
    X[:,i] = le.fit_transform(X[:,i])
#le_y = LabelEncoder()
#y = le.fit_transform(y)  


# In[29]:


X


# In[ ]:





# In[30]:


from sklearn.ensemble import ExtraTreesClassifier


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=100,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)


# In[31]:


indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
sorted_column_names = []
for f in range(X.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print((f + 1, column_names[indices[f]], importances[indices[f]]))
    sorted_column_names.append(column_names[indices[f]])

# Plot the feature importances of the forest
plt.figure(figsize=(25,10))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_column_names,  rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.yscale('log')
plt.ylabel('Importace (Log scaled)')
plt.savefig('feat_importance.pdf', format = 'pdf', dpi=600, bbox_inches='tight')
plt.savefig('feat_importance.png', format = 'png', dpi=600, bbox_inches='tight')

plt.show()


# In[ ]:




