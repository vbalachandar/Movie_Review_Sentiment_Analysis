
from itertools import repeat
import csv
import re
import numpy as np
import nltk
import random
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
from sklearn.metrics import accuracy_score
import gc


word_lists = []
word_list_tagged=[]
posneg_feature_vectors=[]
posneg_feature_vectors_test=[]
posneg_feature_vectors_train = []
posneg_feature_vectors_test = []
full_data=[]
label_vector = []
pos = dict()
neg = dict()
posneg = dict()
part_of_speech=[]
global_index = 0
set_size= 5500
top_k_features = 200
end_index=0
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
            
            
            
            
            #adding all the lines from the training file into the list
            full_data.append(_expand_clitics(row[2]))
            
                                        
           

   
    return 0



#Function to extract features from the training data
def _build_features_train(train_data):
    
    global posneg
    word_list=[]
    word_list_tagged=[]
    for data in train_data:
        #building  features for other parts of speech
        tagged = _tag_input_sentence(data)
        word_lists.append(data)
        word_list_tagged.append(tagged)
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
                   

    # Adding words that are not present in the posneg dictionary
    #please comment out this section if you must use more than 7000 samples
    global global_index
    
    
    
    #Resetting the global index to one more than the length of posneg dictionary
    global_index=len(posneg)+1
    
   
    
    #dictionary to hold words from the data
    word_dict=dict()
    
    
    for word in part_of_speech:
        word_dict[word]=global_index
        global_index=global_index+1
    
   
    #
    posneg=dict(posneg.items()+word_dict.items())
    





#function to preprocess the test data
def _build_features_test(test_data):
    
    
    word_lists = []
    word_list_tagged=[]
    
    
    for data in test_data:
        #building  features for other parts of speech
        tagged = _tag_input_sentence(data)
        word_lists.append(data)
        word_list_tagged.append(tagged)
        
                   
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




#build the representation of the training data
def build_train_data_matrix(size):

    for j in range(size):
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


#building the representation of the test data
def build_test_data_matrix(size):
    vect=[]
    
    for j in range(size):
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
                
        posneg_feature_vectors_test.append(vect)
        

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
            posneg[negatives[k].rstrip('\n')] = dictionary_index
            dictionary_index = dictionary_index+1
    
    
    

    posneg['unknown']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
    posneg['dc']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
        
    posneg['DT']=dictionary_index
    
    dictionary_index=dictionary_index+1
    
    end_index=dictionary_index
    
    
    
    global global_index
    
    global_index = dictionary_index
    
    print "posneg len",len(posneg)
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

def featureSelect(X,y,X_test):
    ch2=SelectKBest(chi2,k=top_k_features)
    X_train=ch2.fit_transform(X, y)
    test=ch2.transform(X_test)
    return X_train,test

def k_fold_cross_validation(train_set,test_set,train_label,test_label):
    clf = svm.SVC(kernel='linear', C=1)
    #clf= linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
    clf.fit(train_set,train_label)
    pred_label=clf.predict(test_set)
    print "Accuracy",accuracy_score(test_label, pred_label)
    return 0








val = _read_data('train.tsv')
_create_pos_neg_features()

random.shuffle(full_data)

num_folds = 5


subset_size = len(full_data)/num_folds

for i in range(num_folds):
    
    testing=full_data[i*subset_size:][:subset_size]
    
    training=full_data[:i*subset_size] + full_data[(i+1)*subset_size:]
        
    print "buliding fv from training"
    
    _build_features_train(training)
    
    build_train_data_matrix(len(training))
       
    
    print "buliding fv from testing"
    _build_features_test(testing)
    
    build_test_data_matrix(len(testing))
    
    
    
    #Extracting class labels for the training set
    new_label = label_vector[:i*subset_size]+label_vector[(i+1)*subset_size:]
    new_label_matrix = np.array(new_label)
    shape = (len(training),1)
    new_label_matrix.reshape(shape)
    
    
    # Extracting class labels for the testing set
    test_label = label_vector[i*subset_size:][:subset_size]
    test_label_matrix = np.array(test_label)
    shape = (len(testing),1)
    test_label_matrix.reshape(shape)
    
    
    
    
    new_feature_vector = np.array(posneg_feature_vectors)
    shape2 = (len(training),len(new_feature_vector[0]))
    new_feature_vector.reshape(shape2)
    
    
    test_feature_vector = np.array(posneg_feature_vectors_test)
    shape3 = (len(testing),len(test_feature_vector[0]))
    test_feature_vector.reshape(shape3)
    
    
    
    


    train_set,test_set = featureSelect(new_feature_vector,new_label_matrix,test_feature_vector)
    k_fold_cross_validation(train_set,test_set,new_label_matrix,test_label_matrix)
    
    
    
    #Setting the feature vector variables to empty at 
    posneg_feature_vectors=[]
    posneg_feature_vectors_test=[]
    
    del new_feature_vector
    del test_feature_vector
    
    
    










