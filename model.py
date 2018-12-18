
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
model = keras.models.load_model('model_k.h5')

for it in range(3):
    for index in range(6):
        data = []
        with open("./train/train1M_%d.csv"%(index+1)) as f:
            for line in f:
                piece = eval(line)
                data.append(piece)

        data = np.array(data)

        x = np.array([ grid_one(piece[:-1].reshape(4,4)) for piece in data ])
        y = keras.utils.to_categorical(data[:,-1], 4)

        sep = 1000000
        x_train = x[:sep]
        x_test = x[sep:]
        y_train = y[:sep]
        y_test = y[sep:]

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=10)
        
        model.save('model_k.h5')

