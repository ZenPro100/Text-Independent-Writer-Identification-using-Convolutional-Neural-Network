# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:08:42 2019

@author: Krishna Chandra
"""

import h5py
import numpy as np
from keras import backend as K
import keras
from tensorflow import keras 
from keras import layers
from keras.models import Model, load_model,Sequential
from keras.layers import Conv2D, SeparableConv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Activation, Input ,BatchNormalization ,Dense ,Dropout

from Preprocessing import load_train, load_test
import Model 
import plots

# Declaring the path
path = 'G:/Internship/Kannada_Dataset'
# Loading the Trainig dataset
X_train, Y_train = load_train(path)

input_shape = X_train.shape 
learning_rate = 1e-4

optimizer = Adam(lr=learning_rate)

# Initializing the model
model = Model.model(input_shape)
model.compile(optimizer = optimizer,  loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary() 
 
# Training the model
history = model.fit(X_train, Y_train, batch_size = 32, epochs = 30, initial_epoch = 0, validation_split = 0.15, verbose=1)

# Decreasing the Learning manually and compling the model again
optimizer = Adam(lr=1e-6)
model.compile(optimizer = optimizer,  loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the model for more 20 epochs
history1 = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=50,initial_epoch = 30, verbose=1, validation_split=0.1)

# Loading the Testing dataset
X_test, Y_test = load_test(path)

# Evaluating the model on Training Dataset
model.evaluate(x=X_train, y=Y_train, batch_size=32 , verbose=1 )

model.evaluate(x=X_test, y=Y_test, batch_size=32 , verbose=1 )

model.save('final_model.h5', model)

# Extracting the Loss and Accuracy for each epoch
loss = history.history['loss']
accuracy = history.history['accuracy']

val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

loss += history1.history['loss']
accuracy += history1.history['accuracy']

val_loss += history1.history['val_loss']
val_accuracy += history1.history['val_accuracy']

loss_arr = np.array(loss, dtype = 'float32')
accuracy_arr = np.array(accuracy, dtype = 'float32')
val_loss_arr = np.array(val_loss, dtype = 'float32')
val_accuracy_arr = np.array(val_accuracy, dtype = 'float32')

# Plotting Graphs
plots.Accuarcy_Graph(history)
plots.Loss_Graph(history)
