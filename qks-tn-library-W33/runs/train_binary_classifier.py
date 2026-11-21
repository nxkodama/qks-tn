#!/usr/bin/env python
# coding: utf-8

# ### Initialization

# In[1]:


import numpy as np

import sys, os, time
gen_fn_dir = os.path.abspath('./common_functions')
sys.path.append(gen_fn_dir)

import qks_tn as qksTN
from general_functions import deskewAll

import tensorflow as tf
from tensorflow_addons.optimizers import AdamW
import keras
tf.get_logger().setLevel('ERROR')

import tensornetwork as tn
tn.set_default_backend('tensorflow')

from keras.datasets import mnist


# ### Loading and pre-processing the data

# In[2]:


# Loading MNIST Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Selecting out (3,5)-MNIST
X_train = X_train[(y_train==3) | (y_train==5)]
y_train = y_train[(y_train==3) | (y_train==5)]
X_test = X_test[(y_test==3) | (y_test==5)]
y_test = y_test[(y_test==3) | (y_test==5)]

y_train[y_train==3] = 0
y_train[y_train==5] = 1
y_test[y_test==3] = 0
y_test[y_test==5] = 1

# Reshaping (2D --> 1D) and rescaling (0-255 --> 0-1) 
X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))
X_train, X_test = X_train/255, X_test/255

# Reducing the precision of the data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Deskewing the data
X_train = deskewAll(X_train)
X_test = deskewAll(X_test)


# ### Defining parameters for QKS and TTN

# In[3]:


# Parameters for QKS
nepisodes = 128
p = 784
q = 1
r = int(p/q)

sigma = 0.125

# Parameters for TTN
chi = 4
nqubits = int(np.log2(chi))
nlayers = int(np.log2(nepisodes/nqubits))


# ### Getting the contraction path

# In[4]:


# Getting contraction path
QKS = qksTN.FeatureEncodingLayer(nepisodes,chi,p,sigma)
rho_test = QKS.call(X_test).numpy()

uni_array = tf.constant(np.zeros(([chi]*4+[nlayers])),dtype='complex64')
obs = []
for j in range(int(nepisodes/2)):
    obs.append(rho_test[0,:,:,j])

nodes_set, edge_order = qksTN.construct_dttn(chi,uni_array,obs,nepisodes)
result, path = qksTN.greedy(nodes_set,output_edge_order=edge_order)


# ### Setting up the model and optimizer for training

# In[5]:


# Setting training parameters
start_epoch = 0
nepochs = 15
batch_size = 32

# Defining the model architecture
tn_model = tf.keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(p,)),
        qksTN.FeatureEncodingLayer(nepisodes,chi,p,sigma),
        qksTN.TNLayer(chi,nlayers,path),
        qksTN.ConstMulBinary()
    ]
)

# Scheduling learning rate and weight decay for AdamW optimizer
lr_schedule = tf.optimizers.schedules.CosineDecayRestarts(1e-3,y_train.shape[0]//batch_size)
wd_schedule = tf.optimizers.schedules.CosineDecayRestarts(4e-4,y_train.shape[0]//batch_size)

# Compiling the model
tn_model.compile(loss='sparse_categorical_crossentropy', optimizer=AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule), metrics=['accuracy'])
tn_model.summary()

# Defining how batches are drawn from the datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=y_train.shape[0]).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)


# ### Training the model

# In[6]:


history = tn_model.fit(train_dataset,
                       epochs=nepochs,
                       initial_epoch=start_epoch,
                       verbose=1)


# ### Evaluating the model

# In[7]:


tn_model.evaluate(test_dataset);

