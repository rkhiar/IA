#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:11:10 2020

@author: riad
"""



import numpy as np
import tensorflow as tf



class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self,  word_emb_dim, seq_max_length):
    super(PositionalEncoding, self).__init__()
    self.word_emb_dim=word_emb_dim
    self.seq_max_length = seq_max_length


  def positional_embedding(self, position, word_emb_dim):
    PE = np.zeros((1, word_emb_dim))
    for i in range(word_emb_dim):
        if i % 2 == 0:
            PE[:, i] = np.sin(position / 10000 ** (i / word_emb_dim))
        else:
            PE[:, i] = np.cos(position / 10000 ** ((i - 1) / word_emb_dim))
    return PE



  def call(self, inputs):     
        
    pes = []
    for i in range(self.seq_max_length):
        pes.append(self.positional_embedding(i, self.word_emb_dim))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
        
    # adding positional_encoding to every embeded sequence in the data set
    return inputs + pes

