#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.layers import Dense, Input, BatchNormalization, Dropout
import pandas as pd
import numpy as np
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard,  ReduceLROnPlateau, CSVLogger


# In[36]:


clean_data = pd.read_csv('data/clean_data.tsv', sep='\t', encoding='utf-8')
clean_data = clean_data.astype(str)
print('Total records:', len(clean_data))
y = np.load('data/labels.npy')
X = np.array(clean_data)
print('Total Records: ', X.shape[0])
print('Total Variables/Attributes: ', X.shape[1])


# In[37]:


from sklearn.preprocessing import LabelEncoder
for i in range (X.shape[1]):
    le = LabelEncoder()
    X[:,i] = le.fit_transform(X[:,i])


# In[38]:


#le_y = LabelEncoder()
#y = le_y.fit_transform(y) 


# In[39]:


num_classes = len(np.unique(y))


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)


# In[54]:


ip = Input(shape=(X.shape[1],))
x = Dense(100, activation='relu') (ip)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
out = Dense(1) (x)


# In[55]:


model = Model(inputs=[ip], outputs=[out])

optimizer = optimizers.Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.2, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])


# In[56]:


logCallback = CSVLogger('logs/train_log.csv', separator=',', append=False)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min')
checkpoint = ModelCheckpoint('models/DNN.h5', monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, cooldown=0, min_lr=0.0000000001, verbose=0)

callbacks_list = [logCallback, earlyStopping, reduce_lr, checkpoint]


# In[57]:


model.fit(X_train, y_train,
         validation_data=(X_test, y_test),
                       batch_size=256,
                       epochs=200,
                       verbose=2,
                       shuffle=True,
                       #class_weight = class_weights,
                       callbacks=callbacks_list)


# In[29]:





# In[ ]:




