#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
import urllib
import numpy as np
import pickle
import pandas as pd
import datetime
import numpy as np


# In[2]:


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
X_PATH = '../datasets/Xtrain.pkl';
Y_PATH = '../datasets/Ytrain.pkl';


# In[3]:


def load_data(path):
    return pd.read_pickle(path)


# In[4]:


x_training = load_data(X_PATH)
y_labels = load_data(Y_PATH)

np.shape(x_training)


# In[5]:


from matplotlib import pyplot as plt
plt.imshow(x_training[1]);


# In[6]:


print(x_training[1])


# In[7]:


plt.imshow(y_labels[1]);


# In[8]:


import tensorflow as tf;
from tensorflow import keras; 


# In[9]:


from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test=train_test_split(x_training, y_labels, test_size=.3, random_state = 42)

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


# In[10]:


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


# In[11]:



tf.random.set_seed(42)
np.random.seed(42)

denoising_encoder = keras.models.Sequential([
    keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    keras.layers.Conv2D(1, kernel_size=18, padding="same", activation="relu"),
    keras.layers.MaxPool2D(pool_size=4),
    keras.layers.Flatten(),
    keras.layers.Dense(11, activation="relu"),
])
denoising_encoder.summary()


# In[12]:


denoising_decoder = keras.models.Sequential([
    keras.layers.Dense(14*14 * 1, activation="relu", input_shape=[11]),
    keras.layers.Reshape([14, 14, 1]),
    keras.layers.Conv2DTranspose(filters=1, kernel_size=18, strides=2,
                                 padding="same", activation="sigmoid")
])
denoising_decoder.summary()


# In[13]:


denoising_ae = keras.models.Sequential([denoising_encoder, denoising_decoder])
denoising_ae.summary()


# In[14]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("model4.h5",
                                                save_best_only=True)
denoising_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Nadam(),
                     metrics=["mse"])
history = denoising_ae.fit(X_train, y_train, epochs=40,
                           validation_data=(X_valid, y_valid), batch_size = 64, callbacks=[checkpoint_cb])


# In[15]:


model = keras.models.load_model("model4.h5") #Rollback to best mode


# In[16]:


n_images = 5
new_images = X_test[:n_images]
new_images_denoised = model.predict(new_images)

plt.figure(figsize=(6, n_images * 2))
for index in range(n_images):
    plt.subplot(n_images, 3, index * 3 + 1)
    plt.imshow(new_images[index])
    plt.axis('off')
    if index == 0:
        plt.title("Original")
    plt.subplot(n_images, 3, index * 3 + 3)
    plt.imshow(new_images_denoised[index])
    plt.axis('off')
    if index == 0:
        plt.title("Denoised")
plt.show()

