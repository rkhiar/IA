# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:42:09 2019

@author: csfrrdkr
"""

# Construction of B&W Image
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw

# XML modules
from lxml import etree
import xml.etree.ElementTree as ET

# OS modules
from os import listdir, walk
import re
import shutil
import os



import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#from PreProcessing import Find_Char
from sklearn.model_selection import train_test_split

#Deep Learning Modules
import tensorflow as tf
import keras
from tensorflow.keras import Sequential #MLP class
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Flatten,GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, MaxPool2D, concatenate #Flatten transforms matrix into vector and Dense is Fully Connected Layer
from tensorflow.train import GradientDescentOptimizer #Backprop algo with Gradient Descent Update Rule
from tensorflow.keras.models import Model
import tensorflow.keras.applications as pretrained
#from src.AsadNet import AsadNet



os.chdir('/home/riad/Devs_Python/object_localization')

#Personel modules
from loc_batch_genarator import BatchGenerator

import warnings
warnings.filterwarnings("ignore")





    
#------------------------
# PATHS definition
#------------------------

DATA_PATH='/home/riad/Devs_Python/object_localization/data/'

TRAIN_DATA=DATA_PATH+"train_images/"
TEST_DATA=DATA_PATH+"test_images/"
TRAIN_LABELS=DATA_PATH+'train_annotations/'
TEST_LABELS=DATA_PATH+'test_annotations/'

LISTS_LABEL=DATA_PATH+'lists/'

#------------------------
# Methods definition
#------------------------
def create_data_part(path):
    files_list = []
    for(repository, sub_repository, file) in walk(os.path.join(path)):
        files_list.extend(file)
    return files_list

def elt(root_xml, xml_dict):
    for child in root_xml:
        xml_dict[child.tag]=child.text
        elt(child, xml_dict)
        
def create_lab_part(path):    
    labels_part={}
    for filename in listdir(path):
        labels_dict={}
        tree = ET.parse(path+filename)
        root = tree.getroot()
        elt(root, labels_dict)
        labels_part[filename]=labels_dict
    return labels_part   
    

#------------------------
# Partitions creation 
#------------------------
data_part={}
data_part['train']=create_data_part(TRAIN_DATA)
data_part['test']=create_data_part(TEST_DATA)


#---------------------------
# Loading labels
#-------------------------- 
labels_part={}
labels_part['train']=create_lab_part(TRAIN_LABELS)
labels_part['test']=create_lab_part(TEST_LABELS)




#------------------------
# Model
#-------------------------- 

# Parameters
params = {'batch_size': 50,
          #'n_classes': 2,
          'img_height': 256,
          'img_width': 256,
          'n_channels': 3,
          'shuffle': True}


# Generators
training_generator = BatchGenerator(TRAIN_DATA, data_part['train'], labels_part['train'], **params)
validation_generator = BatchGenerator(TEST_DATA, data_part['test'], labels_part['test'], **params)





# Model

"""
Inception = pretrained.InceptionV3(weights='imagenet',include_top=False)
x = Inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(4, activation=tf.keras.activations.linear)(x)
model = Model(inputs = Inception.inputs,outputs=out)
for layer in Inception.layers:
    layer.trainable = False
	
"""


# Model
inputs_img = Input(shape=(256,256,3))
conv_layer_01=Conv2D(filters = 16, kernel_size = (50,50), padding = 'valid', activation = 'relu')(inputs_img)
conv_layer_02=Conv2D(filters = 8, kernel_size = (50,50), padding = 'valid', activation = 'relu')(conv_layer_01)
max_pool = MaxPooling2D(pool_size=(4, 4), strides=None, padding="valid")(conv_layer_01)
avg_pool = AveragePooling2D(pool_size=(3, 3), strides=None, padding="valid")(max_pool)
outputs=Flatten()(Conv2D(filters = 4, kernel_size = (17,17), padding = 'valid', activation = 'relu')(avg_pool))
model = Model(inputs=inputs_img, outputs=outputs)

model.compile(loss = tf.keras.losses.MSE, 
                           optimizer = tf.keras.optimizers.Adam(lr=0.0001)
                           , metrics = ['MeanSquaredError'])
   
model.fit_generator(generator=training_generator
                    ,validation_data=validation_generator, epochs = 3
                    #,use_multiprocessing=True,
                    # workers=6
                    )
   
model.save(DATA_PATH+'work/best_model.h5') 
saved_model = tf.keras.models.load_model(DATA_PATH+'work/best_model.h5')
saved_model.summary()
    
  
