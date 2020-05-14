# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
import string
from string import punctuation
from os import listdir
from collections import Counter


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
def clean_doc(doc, vocab):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	review = ' '.join(tokens)
	return review

 
def process_docs(directory, vocab, train_only):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if train_only and filename.startswith('cv9'):
			continue
		if not train_only and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		review = clean_doc(doc, vocab)
		# add to list
		documents.append(review)
	return documents


def max_lenght(directory, vocab, max_length):
    
    for filename in listdir(directory):
        # create the full path of the file to open
        path = directory + '/' + filename
		# load the doc
        doc = load_doc(path)
		# clean doc
        review = clean_doc(doc, vocab)
        
        if len(review.split()) > max_length:
            max_length=len(review.split())
             
    return max_length