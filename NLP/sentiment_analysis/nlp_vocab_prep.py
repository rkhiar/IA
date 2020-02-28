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
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
	# remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
    # update counts
	vocab.update(tokens)


# load all docs in a directory
def process_docs(data_path, train_only, vocab):
    # walk through all files in the folder
    for filename in listdir(data_path):
        # skip any reviews in the test set
        if train_only and filename.startswith('cv9'):
            continue
        if not train_only and not filename.startswith('cv9'):
            continue
		# create the full path of the file to open
        path_file = data_path + '/' + filename
        # add doc to vocab
        add_doc_to_vocab(path_file, vocab)

    
# keep tokens with a min occurrence
def main_tokens(vocab, min_occurance):
    tokens = [k for k,c in vocab.items() if c >= min_occurance]
    return tokens
    	
# save vocab to file
def save_list(out_vocab_path, vocab, min_occurance):
	# convert lines to a single blob of text
	data = '\n'.join(main_tokens(vocab, min_occurance))
	# open file
	file = open(out_vocab_path+'vocab.txt', 'w')
	# write text
	file.write(data)
	# close file
	file.close()        



    

