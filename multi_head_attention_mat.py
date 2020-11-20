#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:09:27 2020

@author: riad
"""

import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, word_emb_dim, nb_heads):
        super(MultiHeadAttention, self).__init__()
        self.key_size = word_emb_dim // nb_heads
        self.nb_heads = nb_heads
        self.word_emb_dim = word_emb_dim
        self.wq = tf.keras.layers.Dense(word_emb_dim) #[tf.keras.layers.Dense(key_size) for _ in range(nb_heads)]
        self.wk = tf.keras.layers.Dense(word_emb_dim) #[tf.keras.layers.Dense(key_size) for _ in range(nb_heads)]
        self.wv = tf.keras.layers.Dense(word_emb_dim) #[tf.keras.layers.Dense(value_size) for _ in range(nb_heads)]
        self.wo = tf.keras.layers.Dense(word_emb_dim)
    
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.nb_heads, self.key_size])
        x = tf.transpose(x, [0, 2, 1, 3])
        return x
        
        
    def call(self, decoder_input, encoder_output, mask=None):
        query = self.wq(decoder_input)
        key = self.wk(encoder_output)
        value = self.wv(encoder_output)
                
        # Split for multihead attention
        batch_size = tf.shape(query)[0]
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        score = tf.matmul(query, key, transpose_b=True)
        score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))

        if mask is not None:                      
            # Apply Padding_mask or look_forward_mask
            score *= mask
            # replace zeros values by -1e9 in order to reduce the exp in next step 
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

        alignment = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(alignment, value)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.key_size * self.nb_heads])
 
        heads = self.wo(context)
        # heads has shape (batch, decoder_len, word_emb_dim)
        return heads
    

'''
import numpy as np



query = np.array([
                 [ [ 1,2,3,4,5,6,7,8 ] ,[ 1,2,3,4,5,6,7,8 ],[ 0,0,0,0,0,0,0,0 ] ], 
                 [ [ 1,2,3,4,5,6,7,8 ] ,[ 1,2,3,4,5,6,7,8 ],[ 1,2,3,4,5,6,7,8 ] ],                
                 [ [ 1,2,3,4,5,6,7,8 ] ,[ 1,2,3,4,5,6,7,8 ],[ 1,2,3,4,5,6,7,8 ] ]
                 ], dtype='float32')   
    

batch_size = query.shape[0]
nb_heads = 4
key_size = 8 // nb_heads
word_emb_dim = 8    


#wq = tf.keras.layers.Dense(word_emb_dim)
#query = wq(query)
query = tf.reshape(query, [batch_size, -1, nb_heads, key_size])    


print(query[0])

#(nb_heads, seq_len, key_size)
query = tf.transpose(query, [0, 2, 1, 3])

print(query[0])

score = tf.matmul(query, query, transpose_b=True)

print(score[0])

score /= tf.math.sqrt(tf.dtypes.cast(key_size, dtype=tf.float32))

alignment = tf.nn.softmax(score, axis=-1)

print(alignment[0])

context = tf.matmul(alignment, query)

# concat all heads after attention mechanism
context = tf.transpose(context, [0, 2, 1, 3])
context = tf.reshape(context, [batch_size, -1, key_size * nb_heads])
 
heads = tf.keras.layers.Dense(word_emb_dim)(context)

print(heads[0])


# masking computation : -------------------------


original_query = np.array([ [1,2,0], [3,4,5], [6,7,8]    ], dtype='float32')   
    

padding_mask = 1 - tf.cast(tf.equal(original_query, 0), dtype=tf.float32)

print(padding_mask)

padding_mask = tf.expand_dims(padding_mask, axis=1)

print(padding_mask)

padding_mask = tf.expand_dims(padding_mask, axis=1)

print(padding_mask)


padding_mask2 = tf.matmul(padding_mask, padding_mask, transpose_a=True)

print(padding_mask2)



# mul between data and emb_mask

print(score)

score *= padding_mask2

print(score)

"""
padding_mask = tf.reshape(padding_mask, [batch_size, -1, nb_heads, key_size])

#(nb_heads, seq_len, key_size)
padding_mask = tf.transpose(padding_mask, [0, 2, 1, 3])

print(padding_mask[0])
"""



#--------------------------------------------

# look ahead masking
        
look_left_only_mask = tf.linalg.band_part(tf.ones((3, 3)), -1, 0)

look_left_only_mask = tf.repeat([look_left_only_mask[0]], repeats=3, axis=0)

print(look_left_only_mask)


mask = tf.expand_dims(look_left_only_mask, axis=1)


print(mask)

mask = tf.expand_dims(mask, axis=1)

print(mask)

mask = tf.matmul(mask, mask, transpose_a=True)

print(mask)



#--------------------------------
'''