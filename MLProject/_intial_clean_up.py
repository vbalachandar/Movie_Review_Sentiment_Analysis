
from itertools import repeat
import csv
import re
import numpy as np
import nltk
from itertools import ifilterfalse
from nltk import pos_tag, word_tokenize
from nltk.corpus import treebank
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric


word_lists = []
word_list_tagged=[]
posneg_feature_vectors = []
label_vector = []
pos = dict()
neg = dict()
posneg = dict()
part_of_speech=[]
global_index = 0
set_size= 5500
top_k_features = 200
count=0
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')

def _read_data(file_name):
    """

    :rtype : object
    """
    row_cnt=0;
    with open(file_name, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        index = 0
        for row in tsvin:
            if(row_cnt==set_size):
                break
            
            row_cnt=row_cnt+1
            inner_list = (_expand_clitics(row[2])).split(' ')
            inner_list_tagged=_tag_input_sentence(_expand_clitics(row[2]))
            word_lists.append(inner_list)
            word_list_tagged.append(inner_list_tagged)
            label_vector.append(row[3])
            
            
            #building  features for other parts of speech
            tagged = _tag_input_sentence(_expand_clitics(row[2]))          
             
            
            count=0
            # to get all the words that are Nouns,Verbs,Adj
            for i in tagged:
                
                 
                if i not in posneg and i not in part_of_speech and i not in stopwords:
                        count=count+1
                        if i[1][0]=='N':
                            part_of_speech.append(i[0])
                        if i[1][0]=='J':
                            part_of_speech.append(i[0])
                        if i[1][0]=='V':
                            part_of_speech.append(i[0])
                   
                            
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
    return tagged

def build_train_data_matrix():
    for j in range(set_size):
        vect = [0 for i in range(global_index)]
        for word,tag in zip(word_lists[j],word_list_tagged[j]):
            if word in posneg:
                vect[posneg[word]] = 1
           
            #Adding direction changer as a feature
            elif tag=='IN' or word.lower() in ('but','yet'):
                vect[posneg['dc']]=1
          
            # adding determiner as a feature
            elif tag=="DT":
                vect[posneg['DT']]=1
            
            # words not in the feature vector are considered as unknown
            else:
                vect[posneg['unknown']]=vect[posneg['unknown']]+1
                
        posneg_feature_vectors.append(vect)
        

    return 0


def _create_pos_neg_features():

    dictionary_index = 0
    #with open('/media/New Volume/Acads/UIClinks/Machine Learning/Project/opinion-lexicon-English/positive-words.txt', 'r') as f:
    with open('C:\\Users\\balachandar\\Documents\\GitHub\\Movie_Review_Sentiment_Analysis\\opinion-lexicon-English\\positive-words.txt', 'r') as f:
        positives = f.readlines()[35:]
        for i in range(len(positives)):
            pos[positives[i].rstrip('\n')] = dictionary_index
            posneg[positives[i].rstrip('\n')] = dictionary_index
            dictionary_index = dictionary_index+1
  
  
    #with open('/media/New Volume/Acads/UIClinks/Machine Learning/Project/opinion-lexicon-English/negative-words.txt', 'r') as f:
    with open('C:\\Users\\balachandar\\Documents\\GitHub\\Movie_Review_Sentiment_Analysis\\opinion-lexicon-English\\negative-words.txt', 'r') as f:
        negatives = f.readlines()[35:]
        for k in range(len(negatives)):
            neg[negatives[k].rstrip('\n')] = dictionary_index
            posneg[positives[i].rstrip('\n')] = dictionary_index
            dictionary_index = dictionary_index+1
    
    
    # Adding words that are not present in the posneg dictionary
    #please comment out this section if you must use more than 7000 samples
    for word in part_of_speech:
        posneg[word]=dictionary_index
        dictionary_index=dictionary_index+1

    posneg['unknown']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
    posneg['dc']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
    posneg['dc']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
    posneg['DT']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
    
    
    
    global global_index
    
    global_index = dictionary_index
    
    print len(posneg)
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
        vect.append(negative_feature )

        posneg_feature_vectors.append(vect)

    return 0



def removeStopwords(wordList):
        #remove the stop words from the NLTK Stop words list
        
        wordList[:] = ifilterfalse(lambda i: (i in stopwords) , wordList)
        return wordList

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


    clf = KNeighborsClassifier(5, 'distance', 'auto')
    #clf = svm.SVC(kernel='linear', C=1)#.fit(X_train, y_train)
    #clf= linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    #clf.fit(train_array,label_array)
    #print "X",len(train_array)
    #dec = clf.decision_function([[1]])
    
    #print clf.score(X_test, y_test)
    scores = cross_validation.cross_val_score(clf, train_array, label_array, cv=5)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #print "classifiers",dec.shape[1]
    #print(scores)
    return 0










val = _read_data('train.tsv')

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









