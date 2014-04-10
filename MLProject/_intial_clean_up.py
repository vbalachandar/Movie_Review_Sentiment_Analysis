
from itertools import repeat
import csv
import re
import numpy as np

from itertools import ifilterfalse
from nltk import pos_tag, word_tokenize
from nltk.corpus import treebank
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics


word_lists = []
posneg_feature_vectors = []
label_vector = []
pos = dict()
neg = dict()
posneg = dict()
global_index = 0
set_size= 10000
top_k_features = 200

def _read_data(file_name):
    """

    :rtype : object
    """
    with open(file_name, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        index = 0
        for row in tsvin:
            inner_list = (_expand_clitics(row[2])).split(' ')
            word_lists.append(inner_list)
            label_vector.append(row[3])
            #_tag_input_sentence(_word_list)

    return 0


def _read_sentiment_words(file_name):
    return 0;


def _expand_clitics(input_phrase):
    """

    :rtype : basestring
    """

    # input_phrase = re.sub('(can\'t)',"cannot",input_phrase)
    input_phrase = re.sub('(ca n\'t)', "cannot", input_phrase)  # because it is like this in the training data
    input_phrase = re.sub('(n\'t)', " not", input_phrase)
    return input_phrase


def _create_matrix():
    return 0


def _create_feature_vector():
    return 0


def _tag_input_sentence(input_phrase):
    tagged = pos_tag(word_tokenize(input_phrase))
    #print tagged
    return 0

def build_train_data_matrix():
    for j in range(set_size):
        vect = [0 for i in range(global_index)]
        for word in word_lists[j]:
            if word in posneg:
                vect[posneg[word]] = 1
        posneg_feature_vectors.append(vect)

    return 0


def _create_pos_neg_features():

    dictionary_index = 0
    with open('/media/New Volume/Acads/UIClinks/Machine Learning/Project/opinion-lexicon-English/positive-words.txt', 'r') as f:
        positives = f.readlines()[35:]
        for i in range(len(positives)):
            pos[positives[i].rstrip('\n')] = dictionary_index
            posneg[positives[i].rstrip('\n')] = dictionary_index
            dictionary_index = dictionary_index+1


    with open('/media/New Volume/Acads/UIClinks/Machine Learning/Project/opinion-lexicon-English/negative-words.txt', 'r') as f:
        negatives = f.readlines()[35:]
        for k in range(len(negatives)):
            neg[negatives[k].rstrip('\n')] = dictionary_index
            posneg[positives[i].rstrip('\n')] = dictionary_index
            dictionary_index = dictionary_index+1

    global global_index
    global_index = dictionary_index
    return 0

def _build_features():

    for training_sample in range(len(word_lists)):
        positive_score = 0
        negative_score = 0
        positive_feature = 0
        negative_feature = 0
        vect = []

        for word in word_lists[training_sample]:
            if word in pos:
                positive_score = positive_score+1
            if word in neg:
                negative_score = negative_score + 1

        if((positive_score + negative_score)!= 0):
            positive_feature = positive_score/(positive_score + negative_score);
            negative_feature = negative_score/(negative_score + positive_score);
        vect.append(positive_feature)
        vect.append(negative_feature)

        posneg_feature_vectors.append(vect)

    return 0

def featureSelect(X,y):
    ch2=SelectKBest(chi2,k=top_k_features)
    X_train=ch2.fit_transform(X, y)
    #X_test=ch2.transform(X_test)
    return X_train

def k_fold_cross_validation(train_set,label_matrix):

    train_size = 0.9*set_size
    test_size = 0.1*set_size

    train_array = np.array(train_set)
    shape = (set_size,top_k_features)
    train_array.reshape(shape)

    label_array = np.array(label_matrix)
    shape = (set_size,)
    label_array.reshape(shape)

   # X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_array, label_array, test_size=0.1, random_state=0)
   # X_train = np.array(X_train)
   # shape = (9000,top_k_features)
   # X_train.reshape(shape)

   # y_train = np.array(y_train)
   # shape = (9000,)
   # y_train.reshape(shape)

   # X_test = np.array(X_test)
   # shape = (1000,top_k_features)
   # X_test.reshape(shape)

   # y_test = np.array(y_test)
   # shape = (1000,)
   # y_test.reshape(shape)


    clf = svm.SVC(kernel='linear', C=1)#.fit(X_train, y_train)
    #print clf.score(X_test, y_test)
    scores = cross_validation.cross_val_score(clf, train_array, label_array, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #print(scores)
    return 0


val = _read_data('/media/New Volume/Acads/UIClinks/Machine Learning/Project/train.tsv')
_create_pos_neg_features()
build_train_data_matrix()

new_label = label_vector[0:set_size]

new_label_matrix = np.array(new_label)
shape = (set_size,1)
new_label_matrix.reshape(shape)

new_feature_vector = np.array(posneg_feature_vectors)
shape2 = (set_size,len(new_feature_vector[0]))
new_feature_vector.reshape(shape2)


train_set = featureSelect(new_feature_vector,new_label_matrix)
k_fold_cross_validation(train_set,new_label_matrix)









