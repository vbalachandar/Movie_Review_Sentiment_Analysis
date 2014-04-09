__author__ = 'abhinaya'

from itertools import repeat
import csv
import re
from itertools import ifilterfalse
from nltk import pos_tag, word_tokenize
from nltk.corpus import treebank


def _read_data(file_name):
    """

    :rtype : object
    """
    with open(file_name, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            _word_List = _expand_clitics(row[2])
            _tag_input_sentence(_word_List)

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


val = _read_data('/media/New Volume/Acads/UIClinks/Machine Learning/Project/train.tsv')
print val