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




class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, word_emb_dim, nb_layers, nb_heads, seq_max_length, mask=True):
        super(Decoder, self).__init__()
        self.word_emb_dim = word_emb_dim
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.seq_max_length = seq_max_length
        self.vocab_size = vocab_size
        
        # One Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, word_emb_dim, input_length=seq_max_length, mask_zero=mask)
        
        # positionnal encoding
        self.pos_encode = PositionalEncoding(word_emb_dim, seq_max_length)
        
        # bottom Multi-Head Attention and Normalization layers
        self.self_attention = [MultiHeadAttention(word_emb_dim, nb_heads) for _ in range(nb_layers)]
        self.self_attention_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(nb_layers)]
        
        # mid Multi-Head Attention and Normalization layers
        self.cross_attention = [MultiHeadAttention(word_emb_dim, nb_heads) for _ in range(nb_layers)]
        self.cross_attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(nb_layers)]

        # nb_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(word_emb_dim * 4, activation='relu') for _ in range(nb_layers)]
        self.dense_2 = [tf.keras.layers.Dense(word_emb_dim) for _ in range(nb_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(nb_layers)]
                
        
    def call(self, sequence, encoder_output):
        
        embed_out = self.embedding(sequence)
        
        # Create masks (padding and look ahead)
        embed_mask=tf.cast(embed_out._keras_mask, dtype='float32')
        # split the mask so that it has the same shape as the headed scores in the multi_head_att layer
        embed_mask = tf.expand_dims(embed_mask, axis=1)
        embed_mask = tf.expand_dims(embed_mask, axis=1)            
        embed_mask = tf.matmul(embed_mask, embed_mask, transpose_a=True)
        
        look_left_only_mask = tf.linalg.band_part(tf.ones((self.seq_max_length, self.seq_max_length)), -1, 0)
        #look_left_only_mask = tf.repeat([look_left_only_mask[0]], repeats=embed_mask.shape[0], axis=0)
        
        # positional encoding
        pos_out = self.pos_encode(embed_out)
        
        bot_sub_in = pos_out
        
        for i in range(self.nb_layers):
            
            # BOTTOM MULTIHEAD SUB LAYER           
            bot_sub_out = self.self_attention[i](bot_sub_in, bot_sub_in, look_left_only_mask)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.self_attention_norm[i](bot_sub_out)
                        
            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.cross_attention[i](mid_sub_in, encoder_output, embed_mask)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.cross_attention_norm[i](mid_sub_out)
                        
            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

        return ffn_out     
    
    
 
'''
raw_inputs=[[1,2,3,4,4,5,12,3],[1,1,1,1]]

padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    raw_inputs, padding="post"
)


print(padded_inputs )


#vocab_size, word_emb_dim, input_length=seq_max_length
#encoder_output = tf.keras.layers.Embedding(100, 12, input_length = 8, mask_zero=True)(padded_inputs)

# One Embedding layer
embedding = tf.keras.layers.Embedding(100, 12, input_length=8, mask_zero=True)
        
# positionnal encoding
pos_encode = PositionalEncoding(12, 8)
        
# bottom Multi-Head Attention and Normalization layers
self_attention = [MultiHeadAttention(12, 4) for _ in range(2)]
self_attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(2)]
        
# mid Multi-Head Attention and Normalization layers
cross_attention = [MultiHeadAttention(12, 4) for _ in range(2)]
cross_attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(2)]

        # nb_layers FFN and Normalization layers
dense_1 = [tf.keras.layers.Dense(12 * 4, activation='relu') for _ in range(2)]
dense_2 = [tf.keras.layers.Dense(12) for _ in range(2)]
ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(2)]
        
dense = tf.keras.layers.Dense(100)




        
embed_out = embedding(padded_inputs)
        
# Create masks (padding and look ahead)
embed_mask=tf.cast(embed_out._keras_mask, dtype='float32')
embed_mask = tf.expand_dims(embed_mask, axis=1)
embed_mask = tf.expand_dims(embed_mask, axis=1)            
embed_mask = tf.matmul(embed_mask, embed_mask, transpose_a=True)


look_left_only_mask = tf.linalg.band_part(tf.ones((8, 8)), -1, 0)
look_left_only_mask = tf.repeat([look_left_only_mask[0]], repeats=2, axis=0)
        
        
# positional encoding
pos_out = pos_encode(embed_out)
        
bot_sub_in = pos_out
        
            
# BOTTOM MULTIHEAD SUB LAYER
            
            
bot_sub_out = self_attention[0](bot_sub_in, bot_sub_in, look_left_only_mask)  #!!!!!!!!!! problem


bot_sub_out = bot_sub_in + bot_sub_out
bot_sub_out = self_attention_norm[0](bot_sub_out)
            
            
# MIDDLE MULTIHEAD SUB LAYER
mid_sub_in = bot_sub_out

mid_sub_out = cross_attention[0](mid_sub_in, embed_out, embed_mask)  # it own embeded out is like encoder output
mid_sub_out = mid_sub_out + mid_sub_in
mid_sub_out = cross_attention_norm[0](mid_sub_out)
            
            
# FFN
ffn_in = mid_sub_out

ffn_out = dense_2[0](dense_1[0](ffn_in))
ffn_out = ffn_in + ffn_out
ffn_out = ffn_norm[0](ffn_out)

logits = dense(ffn_out)
'''