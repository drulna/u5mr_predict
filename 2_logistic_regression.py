#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# In[11]:


clean_data = pd.read_csv('data/clean_data.tsv', sep='\t', encoding='utf-8')
clean_data = clean_data.astype(str)
print('Total records:', len(clean_data))


# In[12]:


clean_data.head()


# In[65]:


y = np.load('data/labels.npy')


# In[14]:


X = np.array(clean_data)
print('Total Records: ', X.shape[0])
print('Total Variables/Attributes: ', X.shape[1])


# In[38]:


lr_classifier = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class='multinomial')


# In[16]:


from sklearn.preprocessing import LabelEncoder
for i in range (X.shape[1]):
    le = LabelEncoder()
    X[:,i] = le.fit_transform(X[:,i])


# In[61]:


le_y = LabelEncoder()
y = le.fit_transform(y)  


# In[35]:


from sklearn.model_selection import train_test_split


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


# In[67]:


lr_classifier.fit(X_train,y_train)


# In[68]:


print('Testing Accuracy: ',lr_classifier.score(X_test, y_test))


# In[69]:


predicted_y = lr_classifier.predict(X_test)


# In[73]:


from sklearn.metrics import f1_score, classification_report, confusion_matrix
print('F1-score Macro Average: ', f1_score(y_test, predicted_y, average='macro'))
print('F1-score Micro Average: ', f1_score(y_test, predicted_y, average='micro'))
print('F1-score Weighted Average: ', f1_score(y_test, predicted_y, average='weighted'))



# In[71]:


print(classification_report(y_test, predicted_y))


# # ROC Curve and AUC cannot be calculated on multi-class or regression problems. So it is necessary to binarize the output

# In[75]:


cm = confusion_matrix(y_test, predicted_y)


# In[84]:


fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tp = np.diag(cm)
tn = cm.sum() - (fp + fn + tp)


# In[88]:


# Recall, Sensitivity, Hit Rate, TPR

tpr = tp/(tp+fn)


# In[90]:


print('TPR/Sensitivity for each class', tpr)


# In[91]:


# Specifitivity, TNR

tnr = tn/(tn+fp)


# In[92]:


print('TNR/Specifitivity', tnr)


# In[ ]:




