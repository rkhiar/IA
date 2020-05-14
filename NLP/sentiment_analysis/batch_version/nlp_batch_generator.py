#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:33:55 2020

@author: riad
"""


from tensorflow import keras
import numpy as np
from nltk.corpus import stopwords
import string
from os import listdir, walk, path
from collections import Counter
from keras.preprocessing.sequence import pad_sequences


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, path, list_ids, labels, max_len_review, vocab, tokenizer, batch_size=32 , n_classes=2, shuffle=True):
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path=path
        self.vocab=vocab
        self.max_len_review=max_len_review
        self.tokenizer=tokenizer
    

    # load doc into memory
    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text


    # turn a doc into clean tokens
    def clean_doc(self, doc, vocab):
        work_review=[]
        # split into tokens by white space
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        work_review.append(' '.join(tokens))
        return work_review
    
     
    def process_docs(self, directory, vocab, batch_file_id):
                # walk through all files in the folder
        # r=root, d=directories, f = files
        for r, d, f in walk(directory):
            for file in f:
                # load the docs in bach files list
                if file == batch_file_id:
                    # load doc
                   doc = self.load_doc(path.join(r, file))
                   # clean doc
                   review = self.clean_doc(doc, vocab)
                           
        return review
    
    
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids) / self.batch_size))
     
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)        
        
    
    def __getitem__(self, idx_start):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_idx = self.indexes[idx_start*self.batch_size:(idx_start+1)*self.batch_size]
        
        # Find list of IDs
        list_ids_batch = [self.list_ids[k] for k in batch_idx]
        
        # Generate data
        X, y = self.__data_generation(list_ids_batch)
        
        return X, y
    
    
    
    
    def __data_generation(self, list_ids_batch):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty([self.batch_size, self.max_len_review], dtype = np.uint8)
        y = np.empty([self.batch_size,1], dtype = np.int)
              
        # Generate data
        for i, ID in enumerate(list_ids_batch):
            # Store sample
            encoded_docs = self.tokenizer.texts_to_sequences(self.process_docs(self.path, self.vocab, ID))
            X0 = pad_sequences(encoded_docs, maxlen=self.max_len_review, padding='post')
           
            X[i,] = np.array(X0)
            # Store class
            y[i] = self.labels[ID]
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)