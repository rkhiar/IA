# -*- coding: utf-8 -*-


from nltk.corpus import stopwords
import string
from string import punctuation
from os import listdir
from collections import Counter
import numpy as np


"""The word2vec algorithm processes documents sentence by sentence. 
This means we will preserve the sentence-based structure during cleaning."""
	

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 


# turn a doc into clean tokens
def doc_to_clean_lines(doc, vocab):
    """returns a list of cleaned lines"""
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines



# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    """ returnes a list (all docs) of lists (doc) of lists (cleaned lines)"""
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines



	
# load embedding as a dict
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = np.array(parts[1:], dtype='float32')
	return embedding


def get_weight_matrix(embedding, vocab_tokenizer):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab_tokenizer) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab_tokenizer.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix