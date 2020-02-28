# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:42:09 2019

@author: csfrrdkr
"""



import os
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



import warnings
warnings.filterwarnings("ignore")


from batch_genarator_rkh import BatchGenerator

    
# Variables definition
train_path="/home/riad/Devs_Python/Credit_Safe/CNN/data/train/"
test_path="/home/riad/Devs_Python/Credit_Safe/CNN/data/test/"


#Methods definition
def CreatePartition(path):
    FilesList = []
    for(repository, sub_repository, file) in walk(os.path.join(path)):
        FilesList.extend(file)
    return FilesList




# Program

partition={}

partition['train']=CreatePartition(train_path)
partition['test']=CreatePartition(test_path)


'''
train_label1=np.hstack((np.zeros((12500,1), dtype=np.int), np.ones((12500,1), dtype=np.int)))
train_label2=np.hstack((np.ones((12500,1), dtype=np.int), np.zeros((12500,1), dtype=np.int)))
train_output=np.vstack((train_label1, train_label2)).tolist()

test_label1=np.hstack((np.zeros((25,1), dtype=np.int), np.ones((25,1), dtype=np.int)))
test_label2=np.hstack((np.ones((25,1), dtype=np.int), np.zeros((25,1), dtype=np.int)))
test_output=np.vstack((test_label1, test_label2)).tolist()
'''

'''
train_labels=dict(zip(partition['train'],train_output))
test_labels=dict(zip(partition['test'],test_output))
'''



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
training_generator = BatchGenerator(train_path, partition['train'], train_labels, **params)
validation_generator = BatchGenerator(test_path, partition['test'], test_labels, **params)


'''
inputs_img = Input(shape=(256,256,3))
flat_input=Flatten()(inputs_img)
x = Dense(200, activation=tf.nn.relu)(flat_input)
x2 = Dense(50, activation=tf.nn.relu)(x)
outputs = Dense(2, activation=tf.nn.softmax)(x2) 
model = Model(inputs=inputs_img, outputs=outputs)
'''

inputs_img = Input(shape=(256,256,3))
conv_layer=Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', activation = 'relu')(inputs_img)
print(conv_layer.shape)
flat_input=Flatten()(conv_layer)
x = Dense(200, activation=tf.nn.relu)(flat_input)
x2 = Dense(50, activation=tf.nn.relu)(x)

outputs = Dense(2, activation=tf.nn.softmax)(x2)
model = Model(inputs=inputs_img, outputs=outputs)


model.compile(loss = "categorical_crossentropy", 
                           optimizer = keras.optimizers.Adam(lr=0.000001)
                           , metrics = ['acc'])
   

model.fit_generator(generator=training_generator
                    ,validation_data=validation_generator, epochs = 50
                    #,use_multiprocessing=True,
                    # workers=6
                    )
   
print("c est finiiiiiiiiiiiiiiiiiiiiiiii")
