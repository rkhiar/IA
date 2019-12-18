# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:20:46 2019

@author: csfrrdkr
"""


from tensorflow import keras
import numpy as np
from PIL import Image

class BatchGenerator(keras.utils.Sequence):
    def __init__(self, path, list_IDs, labels,  batch_size=32, img_height=256, img_width=256, n_channels=3 , n_classes=2, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.img_height = img_height
        self.img_width = img_width
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path=path
        
    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)
            
    

    def ImgResize(self, ImageToResize, width, height):
        img = Image.open(ImageToResize) # image extension *.png,*.jpg
        new_width  = width
        new_height = height
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        #img.save(imSave) # format may what u want ,*.png,*jpg,*.gif
        return img
     
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        
        
        
    
    def __getitem__(self, idx_start):
        'Generate one batch of data'
        # Generate indexes of the batch
        Batch_idx = list(range(idx_start*self.batch_size,(idx_start+1)*self.batch_size))
        
        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in Batch_idx]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_batch)
        
        return X, y
    
    
    
    
    def __data_generation(self, list_IDs_batch):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size, self.img_height, self.img_width, self.n_channels], dtype = np.uint8)
        y = np.empty([self.batch_size,1], dtype = np.int)
                        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            # Store sample
            var=self.path+ID
            X[i,] = np.array(self.ImgResize(var, self.img_height, self.img_width))
            
            # Store class
            y[i] = self.labels[ID]
            
           
        
        return X/255, keras.utils.to_categorical(y, num_classes=self.n_classes)