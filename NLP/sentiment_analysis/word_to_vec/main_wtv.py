# -*- coding: utf-8 -*-



import os
os.chdir('/home/riad/Devs_Python/NLP')


from string import punctuation
from os import listdir
from gensim.models import Word2Vec
import  nlp_wtv_prep as ntp 
from keras.preprocessing.text import Tokenizer
import  nlp_data_prep as ndp 
import numpy as np

#Deep Learning Modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential #MLP class
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Input, Reshape, Concatenate, Bidirectional, TimeDistributed, Conv2D
from tensorflow.train import GradientDescentOptimizer 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

 
DATA_PATH='/home/riad/Devs_Python/NLP/data/movie_review/txt_sentoken/'
WORK_PATH='/home/riad/Devs_Python/NLP/work/'



 
# load the vocabulary
vocab_filename = WORK_PATH+'vocab.txt'
vocab = ntp.load_doc(vocab_filename)
vocab = vocab.split()
set_vocab = set(vocab)
 

#------------------------- Word2vect Embedding--------------------
# load training data
positive_docs = ntp.process_docs(DATA_PATH+'pos', vocab, True)
negative_docs = ntp.process_docs(DATA_PATH+'neg', vocab, True)
sentences = negative_docs + positive_docs
print('Total training sentences: %d' % len(sentences))
 
# train word2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
# summarize vocabulary size in model
words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
 
# save model in ASCII (word2vec) format
filename = WORK_PATH+'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

#-------------------------------------------------------------------


# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(vocab)
#tokenizer.fit_on_sequences(vocab)
    
# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1




# load all training reviews
positive_docs = ndp.process_docs(DATA_PATH+'pos', set_vocab, True)
negative_docs = ndp.process_docs(DATA_PATH+'neg', set_vocab, True)
train_docs = negative_docs + positive_docs

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)

max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = np.array([0 for _ in range(900)] + [1 for _ in range(900)])


positive_docs = ndp.process_docs(DATA_PATH+'pos', set_vocab, False)
negative_docs = ndp.process_docs(DATA_PATH+'neg', set_vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = np.array([0 for _ in range(100)] + [1 for _ in range(100)])



# load embedding from file
raw_embedding = ntp.load_embedding(WORK_PATH+'embedding_word2vec.txt')
# get vectors in the right order
embedding_vectors = ntp.get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], 
                            input_length=max_length, trainable=False)




#----------------------------------------------------
# Model
#----------------------------------------------------


inputs_review = Input(shape = (1317,))
embedding_layer=embedding_layer (inputs_review)
#flatt_embedding=Flatten()(embedding_layer)
#out_embedding= TimeDistributed(Dense(1317, activation=tf.nn.relu))(embedding_layer)#
bi_lstm=LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_layer)
bi_lstm2=LSTM(56, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)(bi_lstm)
#bi_lstm=Bidirectional(LSTM(128, return_sequences=False))(embedding_layer)
#flat_input=Flatten()(bi_lstm)
outputs = Dense(2, activation=tf.nn.softmax)(bi_lstm2)
model = Model(inputs=inputs_review, outputs=outputs)


model.compile(loss = "sparse_categorical_crossentropy", 
                           optimizer = keras.optimizers.Adam(lr=0.0001)
                           , metrics = ['acc'])
 

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint(WORK_PATH+'best_model.h5', monitor='val_loss', mode='min', save_best_only=False)
    
model.fit(Xtrain, ytrain, epochs = 5, batch_size=32, verbose = 1,
                  shuffle=True ,validation_data=(Xtest, ytest)
                  ,callbacks=[es, mc]
                  )  