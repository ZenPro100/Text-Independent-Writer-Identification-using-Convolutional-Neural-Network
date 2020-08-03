# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:42:14 2020

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


def model(input_shape):
    
    inputs = Input(input_shape)
    
    x = Conv2D(64, (3,3), strides=(1,1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1) (x)
    
    x = Conv2D(64, (3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1) (x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    
    
    
    x2 = Conv2D(128, (3,3), strides=(1,1), padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(0.1) (x2)
    
    x2 = Conv2D(128, (3,3), strides=(1,1), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(0.1) (x2)
    x2 = MaxPooling2D((2,2), strides=(2,2))(x2)
    
    
    x4 = Conv2D(256, (3,3), strides=(1,1), padding='same')(x2)
    x4 = BatchNormalization()(x4)
    x4 = LeakyReLU(0.1) (x4)
    
    x4 = Conv2D(256, (3,3), strides=(1,1), padding='same')(x4)
    x4 = BatchNormalization()(x4)
    x4 = LeakyReLU(0.1) (x4)
    x4 = MaxPooling2D((2,2), strides=(2,2))(x4)
    
    
    x6 = Conv2D(512, (3,3), strides=(1,1), padding='same')(x4)
    x6 = BatchNormalization()(x6)
    x6 = LeakyReLU(0.1) (x6)
    
    x6 = Conv2D(512, (3,3), strides=(1,1), padding='same')(x6)
    x6 = BatchNormalization()(x6)
    x6 = LeakyReLU(0.1) (x6)
    x6 = MaxPooling2D((2,2), strides=(2,2))(x6)
    
    
    x8 = Flatten()(x6)
    
    x8 = Dense(4096, activation=LeakyReLU(0.1))(x8)
    x8 = Dropout(0.5)(x8)
    
    x8 = Dense(1096, activation=LeakyReLU(0.1))(x8)
    x8 = Dropout(0.5)(x8)
    
    outputs = Dense(units=57, activation='softmax') (x8)
    
    model = Model(inputs=[inputs], outputs=[outputs]) 
    
    return model