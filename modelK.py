
# coding: utf-8

# # 2048 Keras

# In[1]:


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate, BatchNormalization, Activation
from keras.optimizers import Adadelta
import numpy as np

BATCH_SIZE = 128
NUM_EPOCHS = 15


# In[2]:


OUT_SHAPE = (4,4)
CAND = 16
#map_table = {2**i : i for i in range(1,CAND)}
#map_table[0] = 0

def grid_one(arr):
    ret = np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)  # shape = (4,4,16)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,arr[r,c]] = 1
    return ret


# In[3]:


import csv
data = []
with open("./train/train1M_2.csv") as f:
    for line in f:
        piece = eval(line)
        data.append(piece)


# In[5]:


data = np.array(data)


# In[9]:


x = np.array([ grid_one(piece[:-1].reshape(4,4)) for piece in data ])
y = keras.utils.to_categorical(data[:,-1], 4)


# In[10]:


sep = 900000
x_train = x[:sep]
x_test = x[sep:]
y_train = y[:sep]
y_test = y[sep:]


# In[11]:


x_test.shape


# In[12]:


model = keras.models.load_model('model_k.h5')


# In[13]:


# train , validation_data=(x_test,y_test)
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=30)


# In[48]:


score_test = model.evaluate(x_test,y_test,verbose=0)
print('Testing loss: %.4f, Testing accuracy: %.2f' % (score_test[0],score_test[1]))


# In[15]:


model.save('model_k.h5')  # creates a HDF5 file 'my_model.h5'

