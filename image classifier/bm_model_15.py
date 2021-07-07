# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:42:09 2019

@author: csfrrdkr
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.chdir('/home/riad/Devs_Python/CNN')
from os import walk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#from PreProcessing import Find_Char
from sklearn.model_selection import train_test_split

#Deep Learning Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential #MLP class
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPool2D, concatenate #Flatten transforms matrix into vector and Dense is Fully Connected Layer
from tensorflow.train import GradientDescentOptimizer #Backprop algo with Gradient Descent Update Rule
from tensorflow.keras.models import Model
#from src.AsadNet import AsadNet

#Personel modules
from batch_genarator import BatchGenerator






    
# Variables definition
TRAIN_PATH="/home/riad/Devs_Python/CNN/data/train/"
TEST_PATH="/home/riad/Devs_Python/CNN/data/test/"


# Methods definition
def create_partition(path):
    files_list = []
    for(repository, sub_repository, file) in walk(os.path.join(path)):
        files_list.extend(file)
    return files_list


# Partitions creation 
partition={}

partition['train']=create_partition(TRAIN_PATH)
partition['test']=create_partition(TEST_PATH)

train_output=np.vstack((np.zeros((12500,1), dtype=np.int),np.ones((12500,1), dtype=np.int))).reshape((1,25000)).tolist()
test_output=np.vstack((np.zeros((25,1)),np.ones((25,1)))).reshape((1,50)).tolist()

train_labels=dict(zip(partition['train'],train_output[0]))
test_labels=dict(zip(partition['test'],test_output[0]))


# Parameters
params = {'batch_size': 50,
          'n_classes': 2,
          'img_height': 256,
          'img_width': 256,
          'n_channels': 3,
          'shuffle': True}



# Generators
training_generator = BatchGenerator(TRAIN_PATH, partition['train'], train_labels, **params)
validation_generator = BatchGenerator(TEST_PATH, partition['test'], test_labels, **params)



# Model
inputs_img = Input(shape=(256,256,3))
conv_layer=Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', activation = 'relu')(inputs_img)
flat_input=Flatten()(conv_layer)
dense_layer_01 = Dense(200, activation=tf.nn.relu)(flat_input)
dense_layer_02 = Dense(50, activation=tf.nn.relu)(dense_layer_01)

outputs = Dense(2, activation=tf.nn.softmax)(dense_layer_02)
model = Model(inputs=inputs_img, outputs=outputs)


model.compile(loss = "categorical_crossentropy", 
                           optimizer = keras.optimizers.Adam(lr=0.00001)
                           , metrics = ['acc'])
   

model.fit_generator(generator=training_generator
                    ,validation_data=validation_generator, epochs = 1
                    #,use_multiprocessing=True,
                    # workers=6
                    )
   
