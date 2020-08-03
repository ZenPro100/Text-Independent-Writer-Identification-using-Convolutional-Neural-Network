# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:23:59 2019

@author: Krishna Chandra
"""


import numpy as np
import matplotlib.pyplot as plt

# Accuracy vs Epoch
def Accuracy_Graph(accuracy_arr, val_accuracy_arr):
    plt.plot(accuracy_arr)
    plt.plot(val_accuracy_arr)
    #plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


# Loss vs Epoch
def Loss_Graph(loss_arr,val_loss_arr):

    plt.plot(loss_arr)
    plt.plot(val_loss_arr)
    #plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()