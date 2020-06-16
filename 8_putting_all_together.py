#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[1]:


# Save micro average values for plotting later

import pickle

infile = open('fpr_bin_lr.pickle','rb')
fpr_lr = pickle.load(infile)
infile.close()


infile = open('tpr_bin_lr.pickle','rb')
tpr_lr = pickle.load(infile)
infile.close()


infile = open('rocauc_lr.pickle','rb')
rocauc_lr = pickle.load(infile)
infile.close()

######################################################


infile = open('fpr_bin_dn.pickle','rb')
fpr_dn = pickle.load(infile)
infile.close()


infile = open('tpr_bin_dn.pickle','rb')
tpr_dn = pickle.load(infile)
infile.close()

infile = open('rocauc_dn.pickle','rb')
rocauc_dn = pickle.load(infile)
infile.close()

###############################################



infile = open('fpr_bin_cnn.pickle','rb')
fpr_cnn = pickle.load(infile)
infile.close()


infile = open('tpr_bin_cnn.pickle','rb')
tpr_cnn = pickle.load(infile)
infile.close()


infile = open('rocauc_cnn.pickle','rb')
rocauc_cnn = pickle.load(infile)
infile.close()

###############################################




infile = open('fpr_bin_cnndnn.pickle','rb')
fpr_cnndnn = pickle.load(infile)
infile.close()


infile = open('tpr_bin_cnndnn.pickle','rb')
tpr_cnndnn = pickle.load(infile)
infile.close()

infile = open('rocauc_cnndnn.pickle','rb')
rocauc_cnndnn = pickle.load(infile)
infile.close()

###############################################


# In[11]:


import matplotlib.pyplot as plt
from itertools import cycle
get_ipython().run_line_magic('matplotlib', 'inline')


lw = 2
plt.figure(figsize = (15,15))
plt.plot(fpr_lr["micro"], tpr_lr["micro"],label='LR (area = {0:0.2f})'''.format(rocauc_lr["micro"]),color='deeppink', linestyle=':', linewidth=3)
plt.plot(fpr_dn["micro"], tpr_dn["micro"],label='DN (area = {0:0.2f})'''.format(rocauc_dn["micro"]),color='purple', linestyle=':', linewidth=3)
plt.plot(fpr_cnn["micro"], tpr_cnn["micro"],label='CNN (area = {0:0.2f})'''.format(rocauc_cnn["micro"]),color='blue', linestyle=':', linewidth=3)
plt.plot(fpr_cnndnn["micro"], tpr_cnndnn["micro"],label='CNN-DN (area = {0:0.2f})'''.format(rocauc_cnndnn["micro"]),color='green', linestyle=':', linewidth=3)

#plt.plot(fpr_lr["macro"], tpr_lr["macro"],label='macro-average ROC curve (area = {0:0.2f})'''.format(rocauc_lr["macro"]),color='navy', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_micro.png', bbox_inches = 'tight', format='png', dpi=400)
plt.show()


# In[ ]:




