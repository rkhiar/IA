#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:23:56 2020

@author: riad
"""

import os
os.chdir('/home/riad/Devs_Python/natural_langage_processing/transformer/chatbot/')
import re
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Input, Reshape, Concatenate, Bidirectional, TimeDistributed, Conv2D
from tensorflow.keras.models import Model
import numpy as np
from transformer import Transformer, CustomSchedule, loss_function, create_mask

import warnings 
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


path_bot = "/home/riad/Devs_Python/natural_langage_processing/transformer/chatbot"
path_to_dataset = os.path.join(path_bot, 'cornell_movie_dialogs_corpus')
path_work = os.path.join(path_bot, 'work')

path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset,
                                           'movie_conversations.txt')



#--------------------------
# Load and preprocess
#--------------------------

# Maximum number of samples to preprocess
MAX_SAMPLES = 50000

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    # remove space a the start / end of the sentence
    sentence = sentence.strip()
  
    return sentence


def load_conversations():
    # dictionary of line id to text
    id2line = {}
    with open(path_to_movie_lines, errors='ignore') as file:
       lines = file.readlines()
       # L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]
        # result is 'L902':'HESLO KJABEJ KJEBFZEZ ...'
        #           'L903':'zefz zefzefzef ...'
               
       
    inputs, outputs = [], []  
  
    with open(path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
        # u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
    for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        # ['L194', 'L195', 'L196', 'L197']
        for i in range(len(conversation) - 1):
            inputs.append(preprocess_sentence(id2line[conversation[i]]))
            outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
            if len(inputs) >= MAX_SAMPLES:
                return inputs, outputs
    return inputs, outputs


questions, answers = load_conversations()


#-------------------------------------
# tokenization / words to sequences
#-------------------------------------

# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2




# Maximum sentence length
MAX_LENGTH = 40


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []
  
    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check tokenized sentence max length
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)
  
    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
    return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)


tokenizer.save_to_file(os.path.join(path_work, 'vocab'))


#-----------------
# model
#-----------------



# loss and learning rate defifition
learning_rate = CustomSchedule(D_MODEL)
'''optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)'''

optimizer = tf.keras.optimizers.Adam()
    
inputs_enc= Input(shape = (None,))
inputs_dec= Input(shape = (None,))
# vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length, dropout, mask
outputs=Transformer(VOCAB_SIZE, D_MODEL, 4, 8, MAX_LENGTH, 0.1, True)(inputs_enc, inputs_dec, True) 

model = Model(inputs=[inputs_enc, inputs_dec] , outputs=outputs)

model.compile(optimizer=optimizer, loss = loss_function, metrics = tf.metrics.SparseCategoricalAccuracy())
    
model.fit(x=[questions, answers], y=answers,  epochs = 1
                  #,batch_size=32, verbose = 1
                  #,shuffle=False 
                  #,validation_data=[enc_input, dec_input]
                  ,validation_split=0.2
                  #,callbacks=[es, mc]
                  )  

model.summary()
model.save(os.path.join(path_work, 'model') )


############# bidouilage #####################################################



seq_test_enc=tf.expand_dims(questions[0],0)
seq_test_dec=tf.expand_dims(answers[0],0)

emb, lh = create_mask(seq_test_dec, 9000, D_MODEL)

y=tf.expand_dims(np.array([0,3,2]),0)

predictions = model(inputs=[seq_test_enc, seq_test_dec], training=False)

predictions = model.predict([seq_test_enc, y])






############# bidouilage #####################################################
"""
saved_model = tf.keras.models.load_model(DATA_PATH+'work/best_model.h5')
    
saved_model.summary()"""
    


def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)



  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # concatenated the predicted_id to the output which is given to the decoder as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)



def predict(sentence):
  prediction = evaluate(sentence)
  predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
  return predicted_sentence




sentence='bonjour le robo'

sentence = preprocess_sentence(sentence)

sentence =  START_TOKEN + tokenizer.encode(sentence) + END_TOKEN

output = tf.expand_dims(START_TOKEN, 0)
