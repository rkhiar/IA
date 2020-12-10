#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:44:39 2020

@author: riad
"""
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    ''' we do not pay attention to batches, the calculations are performed the same 
    for 1 or multiple batches'''

    def __init__(self, word_emb_dim, nb_heads, mask=None, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__()
        self.query_size = word_emb_dim // nb_heads
        self.key_size = word_emb_dim // nb_heads
        self.value_size = word_emb_dim // nb_heads
        self.nb_heads = nb_heads
        self.mask = mask
        self.wq_layers = [tf.keras.layers.Dense(self.query_size) for _ in range(nb_heads)]
        self.wk_layers = [tf.keras.layers.Dense(self.key_size) for _ in range(nb_heads)]
        self.wv_layers = [tf.keras.layers.Dense(self.value_size) for _ in range(nb_heads)]
        self.wo_layer = tf.keras.layers.Dense(word_emb_dim)
        



    def scaled_dot_product_attention(self, query, key, value, num_head, mask=None):
        score = tf.matmul(self.wq_layers[num_head](query), self.wk_layers[num_head](key), transpose_b=True)

        # Here we scale the score as described in the paper
        score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
        # score has shape (batch, query_len, value_len)
        
        # mask must be broadcastable to (batch, query_len, value_len)
        if mask is not None:
            
            '''after_trans_maskito =wq_layers(maskito)

            print(tf.where(tf.equal(after_trans_maskito, 0), tf.zeros_like(after_trans_maskito), after_trans_maskito/after_trans_maskito)  )
            '''
            
            score *= mask
            # asign masked positions to -1e9
            # so that their values after softmax are zeros
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)    
            

        alignment = tf.nn.softmax(score, axis=2)
        #alignment has shape (batch, query_len, value_len)

        head = tf.matmul(alignment, self.wv_layers[num_head](value))
        # head has shape (batch, decoder_len, value_size)

        return head
    
    

    def call(self, inputs):
       
        heads = []
        for i in range(self.nb_heads):            
            heads.append(self.scaled_dot_product_attention(inputs, inputs, inputs, i))
            # inputs have shape (batch, query_len, word_emb_dim)
            # query, key and value == inputs (no difference between them since it is a parralel matrix coputations)

        # Concatenate all the attention heads to get the original  word_emb_dim
        heads = tf.concat(heads, axis=2)
        heads = self.wo_layer(heads)
        # heads has shape (batch, query_len, word_emb_dim)
        return heads       
        
 

    
    
