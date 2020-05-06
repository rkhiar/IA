#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:35:10 2020

@author: riad
"""

import os
from os import walk
from os import listdir

os.chdir('/home/riad/Devs_Python/NLP')

# Text processing Modules
import  nlp_vocab_prep as nvp 
import  nlp_data_prep as ndp 
from collections import Counter
import string
from nltk.corpus import stopwords
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt 

#Deep Learning Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential #MLP class
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Input, Reshape, Concatenate, Bidirectional, TimeDistributed, Conv2D
from tensorflow.train import GradientDescentOptimizer 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model


from nlp_batch_generator import BatchGenerator





    
# Variables definition
review_path="/home/riad/Devs_Python/NLP/data/movie_review/txt_sentoken/"
out_vocab_path='/home/riad/Devs_Python/NLP/work/'



#------------------------------------------
# Building the vocabulary
#------------------------------------------

# define vocab
reviews_vocab = Counter()

# pre process data
nvp.process_docs(review_path+'neg', True, reviews_vocab)
nvp.process_docs(review_path+'pos', True, reviews_vocab)

# save tokens to a vocabulary file
nvp.save_list(out_vocab_path, reviews_vocab, 2)


# load the vocabulary
vocab_filename = out_vocab_path+'vocab.txt'
vocab = nvp.load_doc(vocab_filename)
vocab = vocab.split()
set_vocab = set(vocab)

#----------------------------------------------------------------

#--------------------------------------------
# Tokenizer
#--------------------------------------------

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(vocab)
#tokenizer.fit_on_sequences(vocab)

#----------------------------------------------------------------


#-------------------------------------------
#Methods definition
#-------------------------------------------
def max_lenght(self, directory, vocab, max_length):
        
        for filename in listdir(directory):
           
            # create the full path of the file to open
            path = directory + '/' + filename
    		# load the doc
            doc = self.load_doc(path)
    		# clean doc
            review = self.clean_doc(doc, vocab)
            
            if len(review.split()) > max_length:
                max_length=len(review.split())
                 
        return max_length


def CreatePartition(path, FilesList, train_test):
    for filename in listdir(path):
        # skip any reviews in the test set
        if train_test == 'train' and filename.startswith('cv9'):
            continue
        if not train_test == 'train' and not filename.startswith('cv9'):
            continue
        FilesList.append(filename)
        
        
#-------------------------------------------        
# Partitions definition
#-------------------------------------------
        
train_file_list=[]
CreatePartition(review_path+"pos" , train_file_list, 'train')
CreatePartition(review_path+"neg" , train_file_list , 'train')

test_file_list=[]
CreatePartition(review_path+"pos" , test_file_list, 'test')
CreatePartition(review_path+"neg" , test_file_list , 'test')

partition={}

partition['train']=train_file_list
partition['test']=test_file_list

train_output=np.vstack((np.ones((900,1), dtype=np.int),np.zeros((900,1), dtype=np.int))).reshape((1,1800)).tolist()
test_output=np.vstack((np.ones((100,1)),np.zeros((100,1)))).reshape((1,200)).tolist()

train_labels=dict(zip(partition['train'],train_output[0]))
test_labels=dict(zip(partition['test'],test_output[0]))



# Identify the max lenght of a "processed" review
max_len = max_lenght(review_path+'pos', set_vocab, 0)
max_len = max_lenght(review_path+'neg', set_vocab, max_len)




#----------------------------------------------------
# Model
#----------------------------------------------------

# Parameters
params = {'batch_size': 50,
          'n_classes': 2,
          'shuffle': True}

# Generators
training_generator = BatchGenerator(review_path, partition['train'], train_labels, max_len, set_vocab, tokenizer, **params)
validation_generator = BatchGenerator(review_path, partition['test'], test_labels, max_len, set_vocab, tokenizer, **params)


# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1


inputs_review = Input(shape = (1317,))
embedding_layer=Embedding(vocab_size, 100, input_length=max_len, name='riad')(inputs_review)
bi_lstm=Bidirectional(LSTM(200, return_sequences=False))(embedding_layer)
outputs = Dense(2, activation=tf.nn.softmax)(bi_lstm)
model = Model(inputs=inputs_review, outputs=outputs)


model.compile(loss = "categorical_crossentropy", 
                           optimizer = keras.optimizers.Adam(lr=0.0001)
                           , metrics = ['acc'])
 

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint(out_vocab_path+'best_model.h5', monitor='val_loss', mode='min', save_best_only=False)
    
model.fit_generator(generator=training_generator,
                    validation_data = validation_generator, 
                    epochs = 2 ,callbacks=[es, mc]
                  )  


"""model.fit_generator(generator=training_generator
                    ,validation_data=validation_generator, epochs = 50
                    #,use_multiprocessing=True,
                    # workers=6
                    )"""

plt.figure()
plt.plot(model.history.history['loss'], label = 'train')
plt.plot(model.history.history['val_loss'], label = 'valid')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
             
plt.figure()
plt.plot(model.history.history['acc'], label = 'train')
plt.plot(model.history.history['val_acc'], label = 'valid')
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.legend()
    
   
   
print("c est finiiiiiiiiiiiiiiiiiiiiiiii")
