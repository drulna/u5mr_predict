#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn import svm
from sklearn.metrics import roc_curve, auc

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression


# In[2]:


clean_data = pd.read_csv('data/clean_data.tsv', sep='\t', encoding='utf-8')
clean_data = clean_data.astype(str)
print('Total records:', len(clean_data))


# In[3]:


clean_data.head()


# In[4]:


y = np.load('data/labels.npy')


# In[5]:


X = np.array(clean_data)
print('Total Records: ', X.shape[0])
print('Total Variables/Attributes: ', X.shape[1])


# In[6]:


for i in range (X.shape[1]):
    le = LabelEncoder()
    X[:,i] = le.fit_transform(X[:,i])


# In[7]:


le_y = LabelEncoder()
y = le.fit_transform(y)  


# In[8]:


y = label_binarize(y, classes=np.unique(y))

nbr_classes = y.shape[1]


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


classifier = OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=100, random_state=42))
y_pred = classifier.fit(X_train, y_train).decision_function(X_test)


# In[11]:


y_pred.shape


# In[12]:


y_pred = classifier.decision_function(X_test)


# In[13]:


print('Testing Accuracy: ',classifier.score(X_test, y_test))


# In[14]:


predicted_y = classifier.predict(X_test)


# In[15]:


from sklearn.metrics import f1_score, classification_report, confusion_matrix
print('F1-score Macro Average: ', f1_score(y_test, predicted_y, average='macro'))
print('F1-score Micro Average: ', f1_score(y_test, predicted_y, average='micro'))
print('F1-score Weighted Average: ', f1_score(y_test, predicted_y, average='weighted'))



# In[16]:


print(classification_report(y_test, predicted_y))


# In[17]:


# Compute FPR TPR


# In[18]:


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(nbr_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[19]:


# Compute micro-average ROC and Area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[20]:


# Compute macro-average ROC and Area

# Aggregate all False Positive Rates
from scipy import interp
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(nbr_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
mean_tpr /= nbr_classes
fpr['macro'] = all_fpr
tpr['macro'] = mean_tpr
roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])


# # Plot Time

# In[21]:


import matplotlib.pyplot as plt
from itertools import cycle
get_ipython().run_line_magic('matplotlib', 'inline')
lw = 2
plt.figure(figsize = (25,25))
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nbr_classes), colors):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# # Plot macro AUC

# In[22]:


import matplotlib.pyplot as plt
from itertools import cycle
lw = 2
plt.figure(figsize = (25,25))
plt.plot(fpr["micro"], tpr["micro"],label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[23]:


# Save micro average values for plotting later

import pickle

outfile = open('fpr_bin_lr.pickle','wb')
pickle.dump(fpr,outfile)
outfile.close()

outfile = open('tpr_bin_lr.pickle','wb')
pickle.dump(tpr,outfile)
outfile.close()

outfile = open('rocauc_lr.pickle','wb')
pickle.dump(roc_auc,outfile)
outfile.close()


# # Calculate specifitivity and sensitivity through confusion matrix

# In[52]:


cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(predicted_y, axis=1))


# In[53]:


fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tp = np.diag(cm)
tn = cm.sum() - (fp + fn + tp)


# In[54]:


# Recall, Sensitivity, Hit Rate, TPR

tpr = tp/(tp+fn)


# In[55]:


print('TPR/Sensitivity for each class', tpr)


# In[56]:


# Specifitivity, TNR

tnr = tn/(tn+fp)


# In[57]:


print('TNR/Specifitivity', tnr)


# In[ ]:




