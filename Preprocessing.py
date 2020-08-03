# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:31:57 2020

@author: Krishna Chandra
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
import h5py
from PIL import Image
import keras
from sklearn.utils import shuffle


def load_train(path) :
    X_train = []
    Y_train = []
    
    for k in range(1,4):
        for i in range(1,58):
            if(k==3 and (i>36 and i<40)):
                continue
            if(i<10):
                name = "Kannada_"+str(k)+"_00"+str(i)+".tif"
            else :
                name = "Kannada_"+str(k)+"_0"+str(i)+".tif"
            img = Image. open(path + name)
            #Extracting the images and converting that into array and appending it to the list
            img = np.asarray(img,'float32')
            p0 = img.shape[0] - img.shape[0] % 128
            p1 = img.shape[1] - img.shape[1] % 128
            patches = [img[m : m + 128, j : j + 128] for j in range(0, p1, 128) for m in range(0, p0, 128)]
            for p in range(len(patches)):
                if(np.min(patches[p])!=np.max(patches[p])):
                    c = patches[p]
                    c = c[:,:,np.newaxis]
                    X_train.append(c)
                    Y_train.append(i-1)
    
    X_train = np.array(X_train, dtype = 'float32')
    Y_train = np.array(Y_train, dtype = 'float32') 
    
    Y_train = keras.utils.to_categorical(Y_train, num_classes=57, dtype='float32') 
    
    X_train, Y_train = shuffle(X_train, Y_train, random_state = 32)
    
    return X_train, Y_train
    
def load_test(path):
    X_test = []
    Y_test = []
    k = 4
    for i in range(1,58):
        
        if(i<10):
            name = "Kannada_" + str(k) + "_00" + str(i) + ".tif"
        else:
            name = "Kannada_" + str(k) + "_0" + str(i) + ".tif"
        img = Image.open(path + name)
        img = np.asarray(img,'float32')
        p0 = img.shape[0] - img.shape[0]%128
        p1 = img.shape[1] - img.shape[1]%128
        patches = [img[m:m+128, j:j+128] for j in range(0,p1,128) for m in range(0,p0,128)]
        for p in range(len(patches)):
            if(np.min(patches[p]) != np.max(patches[p])):
                c = patches[p]
                c = c[:,:,np.newaxis]
                X_test.append(c)
                Y_test.append(i-1)
                    
    X_test = np.array(X_test, dtype = 'float32')
    Y_test = np.array(Y_test, dtype = 'float32')
    Y_test = keras.utils.to_categorical(Y_test, num_classes=57, dtype='float32') 
    
    return X_test, Y_test

