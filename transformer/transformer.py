#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:05:24 2020

@author: riad
"""

import os
os.chdir('/home/riad/Devs_Python/natural_langage_processing/transformer/chatbot/')

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, LSTM, Dense, Dropout, Input, Reshape, Concatenate, Bidirectional, TimeDistributed, Conv2D
from tensorflow.keras.models import Model
from encoder import Encoder
from decoder import Decoder
import numpy as np



class Transformer(tf.keras.Model):
  def __init__(self, vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length, dropout, mask=True):
    super(Transformer, self).__init__()
    self.encoder = Encoder(vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length, dropout, mask)
    self.decoder = Decoder(vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length, dropout, mask)
    self.final_layer = tf.keras.layers.Dense(vocab_size, activation='relu', name='output_dense_layer')
        

  def call(self, enc_sequence, dec_sequence, training):

    enc_output = self.encoder(enc_sequence, training)
    dec_output = self.decoder(dec_sequence, enc_output, training)
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output



def create_mask(padded_sequence, vocab_size, word_emb_dim):
    
    seq_length=padded_sequence.shape[1]
    embedding = tf.keras.layers.Embedding(vocab_size, word_emb_dim, input_length=seq_length, mask_zero=True)
    embed_out = embedding(padded_sequence)
    
    # Create masks (padding and look ahead)
    embed_mask=tf.cast(embed_out._keras_mask, dtype='float32')
    
    # split the mask so that it has the same shape as the headed scores in the multi_head_att layer
    embed_mask = tf.expand_dims(embed_mask, axis=1)
    embed_mask = tf.expand_dims(embed_mask, axis=1)            
    embed_mask = tf.matmul(embed_mask, embed_mask, transpose_a=True)
    
    look_left_only_mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
         
    return embed_mask, look_left_only_mask



def loss_function(y_true, y_pred):
            
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')                            
    loss_ = loss_object(y_true, y_pred)

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


'''

if __name__ == '__main__':
    
    enc_input=np.array([
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8]
               ])
    dec_input=np.array([
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8],
               [1,2,3,4,5,0,0,0],
               [1,2,3,4,5,6,7,8]
               ])
    
 

    # vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length
    outputs=Transformer(1000, 12, 1, 4, 8, 0.2, True)(enc_input, dec_input)     
    outputs = Dense(1000, activation=tf.nn.softmax)(outputs)
    print(outputs)
    print(tf.argmax(outputs,-1))
    
    
    
    
    # loss and learning rate defifition
    
    
    learning_rate = CustomSchedule(12)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)
    
    
    
    
        
    inputs_enc= Input(shape = (8,))
    inputs_dec= Input(shape = (8,))
    # vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length
    outputs=Transformer(1000, 12, 1, 4, 8, 0.2, True)(inputs_enc, inputs_dec) 
    
    model = Model(inputs=[inputs_enc, inputs_dec] , outputs=outputs)
    
    model.compile(optimizer=optimizer, loss = loss_function, metrics = tf.metrics.SparseCategoricalAccuracy())
    
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    #mc = ModelCheckpoint(WORK_PATH+'best_model.h5', monitor='val_loss', mode='min', save_best_only=False)
        
    model.fit(x=[enc_input, dec_input], y=dec_input,  epochs = 10
                      #,batch_size=32, verbose = 1
                      #,shuffle=False 
                      #,validation_data=[enc_input, dec_input]
                      ,validation_split=0.15
                      #,callbacks=[es, mc]
                      )  
    
    
    
    
    
    
    #------------------- test loss mask
    
    
    mask = tf.math.logical_not(tf.math.equal(dec_input, 0))
    mask = tf.cast(mask, dtype='float')
    
    print(tf.reduce_sum(mask))
    
    
    #------------------- test cross entropy
    
    
    y_hat=np.array([
                   [0.8,0.1,0.1,0],
                   [0,0,0.1,0.9],
                   [0,0,1,0]
                   ])
    y=np.array([0,3,2])
    
    
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False , reduction='none'
        )
    
    print(loss_object(y, y_hat ).numpy()
    )
'''
