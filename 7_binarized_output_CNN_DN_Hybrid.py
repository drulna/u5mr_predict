#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn.metrics import roc_curve, auc



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


# In[8]:


np.unique(y)


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


# In[10]:


from keras.layers import Dense, Input, BatchNormalization, Dropout, Reshape, Conv1D, GlobalMaxPool1D, concatenate
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard,  ReduceLROnPlateau, CSVLogger




ip = Input(shape=(X.shape[1],))
x = Reshape((-1,X.shape[1]))(ip)

x = Conv1D(filters=128,kernel_size=2,padding='causal',activation='relu',strides=1)(x)
x = GlobalMaxPool1D()(x)


x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

xx = Dense(100, activation='relu') (ip)
xx = BatchNormalization()(xx)
xx = Dropout(0.2)(xx)
xx = Dense(50, activation='relu')(xx)
xx = BatchNormalization()(xx)
xx = Dropout(0.2)(xx)

combined = concatenate([x, xx])

out = Dense(nbr_classes, activation='softmax') (combined)
model = Model(inputs=[ip], outputs=[out])

optimizer = optimizers.Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.2, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[11]:


logCallback = CSVLogger('logs/train_log_CNN.csv', separator=',', append=False)
earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='max')
checkpoint = ModelCheckpoint('models/CNN_DN.h5', monitor='val_acc', save_weights_only=True, verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, cooldown=0, min_lr=0.0000000001, verbose=0)

callbacks_list = [logCallback, earlyStopping, reduce_lr, checkpoint]


# In[12]:


model.fit(X_train, y_train,
         validation_data=(X_test, y_test),
                       batch_size=256,
                       epochs=200,
                       verbose=2,
                       shuffle=True,
                       #class_weight = class_weights,
                       callbacks=callbacks_list)


# In[13]:


from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score


# In[14]:


model.load_weights('models/CNN_DN.h5')


# In[15]:


y_pred = model.predict(X_test)


# In[16]:


label_pred = np.argmax(y_pred, 1)
label_test = np.argmax(y_test, 1)


# In[17]:


print('Testing Accuracy: ',accuracy_score(label_pred, label_test))


# In[18]:


print('F1-score Macro Average: ', f1_score(label_test, label_pred, average='macro'))
print('F1-score Micro Average: ', f1_score(label_test, label_pred, average='micro'))
print('F1-score Weighted Average: ', f1_score(label_test, label_pred, average='weighted'))



# In[19]:


print(classification_report(label_test, label_pred))


# # Compute FPR TPR

# In[20]:


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(nbr_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[21]:


# Compute micro-average ROC and Area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[22]:


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

# In[23]:


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

# In[24]:


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


# In[25]:


# Save micro average values for plotting later

import pickle

outfile = open('fpr_bin_cnndnn.pickle','wb')
pickle.dump(fpr,outfile)
outfile.close()

outfile = open('tpr_bin_cnndnn.pickle','wb')
pickle.dump(tpr,outfile)
outfile.close()

outfile = open('rocauc_cnndnn.pickle','wb')
pickle.dump(roc_auc,outfile)
outfile.close()


# # Calculate specifitivity and sensitivity through confusion matrix

# In[76]:


cm = confusion_matrix(label_test, label_pred)


# In[77]:


fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tp = np.diag(cm)
tn = cm.sum() - (fp + fn + tp)


# In[78]:


# Recall, Sensitivity, Hit Rate, TPR

tpr = tp/(tp+fn)


# In[79]:


print('TPR/Sensitivity for each class', tpr)


# In[80]:


# Specifitivity, TNR

tnr = tn/(tn+fp)


# In[81]:


print('TNR/Specifitivity', tnr)


# In[ ]:





# In[ ]:




