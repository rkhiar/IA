#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:36:10 2020

@author: riad
"""

import tensorflow as tf
import os

os.chdir('/home/riad/Devs_Python/natural_langage_processing/transformer/chatbot/')

from multi_head_attention_mat import MultiHeadAttention
from positional_encoding import PositionalEncoding
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length, dropout, mask=True):
        super(Encoder, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.seq_max_length = seq_max_length
        self.vocab_size = vocab_size
        self.dropout = dropout
        # One Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, word_emb_dim, input_length=seq_max_length, mask_zero=mask)
        
        # positionnal encoding
        self.pos_encode = PositionalEncoding(word_emb_dim, seq_max_length)
        
        # nb_layers Multi-Head Attention and Normalization layers
        self.attention = [MultiHeadAttention(word_emb_dim, nb_heads) for _ in range(nb_layers)]
        #self.attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(nb_layers)]
        self.attention_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(nb_layers)]

        # nb_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(word_emb_dim * 4, activation='relu') for _ in range(nb_layers)]
        self.dense_2 = [tf.keras.layers.Dense(word_emb_dim, activation='relu' ) for _ in range(nb_layers)]
        self.dropout = [tf.keras.layers.Dropout(rate=dropout) for _ in range(nb_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(nb_layers)]
        
             
    def call(self, sequence, training):
        
        embed_out = self.embedding(sequence)
        
        embed_mask=tf.cast(embed_out._keras_mask, dtype='float32')
        # split the mask so that it has the same shape as the headed scores in the multi_head_att layer
        embed_mask = tf.expand_dims(embed_mask, axis=1)
        embed_mask = tf.expand_dims(embed_mask, axis=1)            
        embed_mask = tf.matmul(embed_mask, embed_mask, transpose_a=True)
                
        pos_out = self.pos_encode(embed_out)        
        sub_in = pos_out        
        for i in range(self.nb_layers):
            sub_out = self.attention[i](sub_in, sub_in, embed_mask)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)
            
            ffn_in = sub_out

            ffn_out = self.dropout[i](self.dense_2[i](self.dense_1[i](ffn_in)), training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out
            
        return ffn_out     
 



