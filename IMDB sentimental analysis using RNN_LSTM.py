#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import keras
from keras.datasets import imdb

((XT,YT),(Xt,Yt)) = imdb.load_data(num_words=10000)


# In[7]:


idx_word = dict([value,key] for (key,value) in word_index.items())
actual_review = ' '.join([idx_word.get(idx-3,'?') for idx in XT[0]])
print(actual_review)
print(len(actual_review.split()))


# In[8]:


## create a 2d tensor to be processed by the embedding 
from keras.preprocessing import sequence
X_train=sequence.pad_sequences(XT,maxlen=500)
X_test=sequence.pad_sequences(Xt,maxlen=500)

print(X_train.shape)
print(X_test.shape)


 


# In[9]:


from keras.layers import Embedding,SimpleRNN,Dense
from keras.models import Sequential

model = Sequential()
model.add(Embedding(10000,64))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()


# In[10]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

from keras.callbacks import ModelCheckpoint 
from keras.callbacks import EarlyStopping 

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_acc',patience=3)



hist = model.fit(X_train,YT,validation_split=0.2,epochs=10,batch_size=128,callbacks=[checkpoint,earlystop])


# In[11]:


import matplotlib.pyplot as plt

acc = hist.history['acc']
val_acc = hist.history['val_acc']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1,len(loss)+1)

plt.title("Loss vs Epochs")
plt.plot(epochs,loss,label="Training Loss")
plt.plot(epochs,val_loss,label="Val Loss")
plt.legend()
plt.show()



plt.title("Accuracy vs Epochs")
plt.plot(epochs,acc,label="Training Acc")
plt.plot(epochs,val_acc,label="Val Acc")
plt.legend()
plt.show()


# In[12]:



get_ipython().system('ls')


# In[13]:


model.load_weights("best_model.h5")

model.evaluate(X_test,Yt)[1]


model.evaluate(X_train,YT)


# In[13]:




