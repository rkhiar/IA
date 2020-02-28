# -*- coding: utf-8 -*-

import os
os.chdir('/home/riad/Devs_Python/NLP')


# Text processing Modules
import  nlp_vocab_prep as nvp 
import  nlp_data_prep as ndp 
from collections import Counter
import string
from nltk.corpus import stopwords
from string import punctuation
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


#Deep Learning Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential #MLP class
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Input, Reshape, Concatenate, Bidirectional, TimeDistributed, Conv2D
from tensorflow.train import GradientDescentOptimizer 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


data_path='/home/riad/Devs_Python/NLP/data/movie_review/txt_sentoken/'
out_vocab_path='/home/riad/Devs_Python/NLP/work/'



#------------------------------------------
# Building the vocabulary
#------------------------------------------

# define vocab
reviews_vocab = Counter()

# pre process data
nvp.process_docs(data_path+'neg', True, reviews_vocab)
nvp.process_docs(data_path+'pos', True, reviews_vocab)

# save tokens to a vocabulary file
nvp.save_list(out_vocab_path, reviews_vocab, 2)



#------------------------------------------
# Embedding layer
#------------------------------------------


# load the vocabulary
vocab_filename = out_vocab_path+'vocab.txt'
vocab = nvp.load_doc(vocab_filename)
vocab = vocab.split()
set_vocab = set(vocab)
#print(vocab)

	

# load all training reviews
positive_docs = ndp.process_docs(data_path+'pos', set_vocab, True)
negative_docs = ndp.process_docs(data_path+'neg', set_vocab, True)
train_docs = negative_docs + positive_docs


# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(vocab)
#tokenizer.fit_on_sequences(vocab)
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
#print(encoded_docs)



# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])



positive_docs = ndp.process_docs(data_path+'pos', set_vocab, False)
negative_docs = ndp.process_docs(data_path+'neg', set_vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])



	
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1



#----------------------------------------------------
# Model
#----------------------------------------------------


inputs_review = Input(shape = (1317,))
embedding_layer=Embedding(vocab_size, 100, input_length=max_length)(inputs_review)
flatt_embedding=Flatten()(embedding_layer)
#out_embedding= TimeDistributed(Dense(1317, activation=tf.nn.relu))(embedding_layer)
bi_lstm=Bidirectional(LSTM(128, return_sequences=False))(flatt_embedding)
#flat_input=Flatten()(bi_lstm)
outputs = Dense(2, activation=tf.nn.softmax)(bi_lstm)
model = Model(inputs=inputs_review, outputs=outputs)


model.compile(loss = "sparse_categorical_crossentropy", 
                           optimizer = keras.optimizers.Adam(lr=0.0001)
                           , metrics = ['acc'])
 

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint(out_vocab_path+'best_model.h5', monitor='val_loss', mode='min', save_best_only=False)
    
model.fit(Xtrain, ytrain, epochs = 5, batch_size=128, verbose = 1,
                  shuffle=True ,validation_data=(Xtest, ytest)
                  ,callbacks=[es, mc]
                  )  

   
    
    

"""model.fit_generator(generator=training_generator
                    ,validation_data=validation_generator, epochs = 50
                    #,use_multiprocessing=True,
                    # workers=6
                    )"""
