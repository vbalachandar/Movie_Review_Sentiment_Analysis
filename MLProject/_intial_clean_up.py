__author__ = 'abhinaya'

from itertools import repeat
import csv
import re
from itertools import ifilterfalse
from nltk import pos_tag, word_tokenize
from nltk.corpus import treebank

word_lists = []
posneg_feature_vectors = []

def _read_data(file_name):
    """

    :rtype : object
    """
    with open(file_name, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            _word_list = _expand_clitics(row[2])
	    word_lists.extend(_word_list)
            #_tag_input_sentence(_word_list)

    return 0


def _read_sentiment_words(file_name):

    return 0;

def _expand_clitics(input_phrase):
    """

    :rtype : basestring
    """

   # input_phrase = re.sub('(can\'t)',"cannot",input_phrase)
    input_phrase = re.sub('(ca n\'t)',"cannot",input_phrase) # because it is like this in the training data
    input_phrase = re.sub('(n\'t)', " not", input_phrase)
    return input_phrase



def _create_matrix():
    return 0


def _create_feature_vector() :
    return 0

def _tag_input_sentence(input_phrase):

    tagged = pos_tag(word_tokenize(input_phrase))
    #print tagged
    return 0

def _create_pos_neg_features() :
	with open('../opinion-lexicon-English/positive-words.txt', 'r') as f:
		positives = f.readlines()[35:]
		pos = dict()
		for i in range(len(positives)):
			pos[positives[i]] = i
	with open('../opinion-lexicon-English/negative-words.txt', 'r') as f:
		negatives = f.readlines()[35:]
		neg = dict()
		for i in range(len(positives), len(positives) + len(neg)):
			neg[negatives[i]] = i
	posneg = dict(list(pos.items()) + list(neg.items()))
	for i in range(len(word_lists)):
		vect = [0 for i in range(len(posneg.keys())]
		for word in word_lists[i].split(' '):
			if word in posneg:
				vect[posneg[word]] = 1
        	posneg_feature_vectors.extend(vect)

val = _read_data('../train.tsv')
_create_pos_neg_features()
print posneg_feature_vectors
print val
