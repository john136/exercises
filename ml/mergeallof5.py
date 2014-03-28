######################################task1.py


""" Features


The objective of this task is to explore the corpus, deals.txt. 


The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:


1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?


"""
###################################################
# Solution 1of Q1:
# A clumsy method for term extraction without analysis of NLPS
#
# Other methods:
# Use some existing NLP tools for term extraction
# http://termcoord.wordpress.com/about/testing-of-term-extraction-tools/free-term-extractors/
# from topia.termextract.src.topia.termextract import extract
#
# March 21, 2014
###################################################

openfile = open('..\data\deals.txt', 'r')

# create the dictionary d

d = {}
removewords = ['online', 'lessons', 'learn', 'shop', 'shipping', 'free', 'save','new','code', 'link',
               'who', 'what', 'when', 'where', 'why', 'how','that', 'is', 'can', 'does', 'do', 'get', 'you', 'he', 'they','our','are',
               '&', 'a','all','this', 'and','by','over', 'to', 'the', 'up','off', 'at', 'in', '-','or','from','for', 'on', 'of', 'with', 'your']


for line in openfile:
    terms = line.split(' ')
    
    for term in terms:
        term = term.strip()

        if not term:
            continue
    
        # remove non-terms
        if term.lower() in removewords:
            continue
            
        # update and aggregate dictionary d with term
        if not (term in d):
            d[term] = 0
        d[term] += 1

# the most/the least popular term

v = list(d.values())

#maxkey = k[v.index(max(v))]
maxvalue = max(v)
minvalue = min(v)

maxkeys = []
minkeys = []

for k, v in d.items():
    if v == maxvalue:
        maxkeys.append(k)
        
    if v == minvalue:
        minkeys.append(k)

# output results

print("The most popular terms\n", maxkeys)
print("the least popular terms\n", minkeys)

#    
# 3. How many types of guitars are mentioned across all the deals?
#

openfile = open('..\data\deals.txt', 'r')

guitars = {}
for line in openfile:
    terms = line.split(' ')

    #print(terms)

    if not 'Guitar' in terms:
        continue

    print(terms)

    terms = (x for x in terms if not x in removewords)
    
    if not terms in guitars:
         guitars[terms] = 0
    guitars[terms] +=1

print("Number of types of guitars =", len(guitars))



########################################### end of task1.py

###################################################### task1b.py

""" Features


The objective of this task is to explore the corpus, deals.txt. 


The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:


1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?


"""

####################################################
# Solution 2 of Q1:
# Use topia.termextract 1.1.0 for term extraction
# 
####################################################

# load term extraction library

from topia.termextract import extract
extractor = extract.TermExtractor()

# define the trivial permissive filter
extractor.filter = extract.permissiveFilter

# load data
openfile = open('..\data\deals.txt', 'r')

d = {}
numberguitars = 0

for line in openfile:

    terms = extractor(line)

    # empty
    if not terms:
        continue

    # take each term from terms
    for term in terms:
            
        # aggregate dictionary for each term
        if not (term[0] in d):
            d[term[0]] = 0
        d[term[0]] += term[1]

        # count guitar
        if 'guitar' in term or 'guitars' in term:
            numberguitars += 1
        else:
            if 'Guitar' in term or 'Guitars' in term:
                numberguitars += 1

v = list(d.values())

maxvalue = max(v)
minvalue = min(v)

maxkeys = []
minkeys = []

for k, v in d.items():
    if v == maxvalue:
        maxkeys.append(k)
        
    if v == minvalue:
        minkeys.append(k)

# output results

print "1. the most popular terms\n", maxkeys
#print "2. the least popular terms\n", minkeys
print "3. the number of types of guitars", numberguitars

#############################end of task1b.py


#############################task2.py

""" Groups and Topics


The objective of this task is to explore the structure of the deals.txt file. 


Building on task 1, we now want to start to understand the relationships to help us understand:


1. What groups exist within the deals?
2. What topics exist within the deals?


"""
###########################################
# Solution of Q2:
# Using termextract and tagging tools from topia for term extraction; 
# Ranking for top k popular terms, which are restricted to NN or NNS 
# Results for answering 1 and 2
# Other part of speech tagging methods can be used
###########################################

from topia.termextract import extract
from topia.termextract import tag

tagger = tag.Tagger()
tagger.initialize()

extractor = extract.TermExtractor()
extractor.filter = extract.permissiveFilter

# load data
openfile = open('..\data\deals.txt', 'r')

# define dictionary
d = {}

# iterative process
for line in openfile:

    terms = extractor(line)

    # remove empty
    if not terms:
        continue

    # take each term from terms
    for term in terms:

        tag = tagger(term[0])

        # remove unrelevant terms by tagging
        if not tag[0][1] in ['NN', 'NNS']:
            continue

        #print tag[0][1]
        
        # aggregate dictionary for each term
        if not (term[0] in d):
            d[term[0]] = 0
        d[term[0]] += term[1]

# sorting for ranking topics
term_tuples = d.items()
term_tuples = sorted(term_tuples, key = lambda term: term[1], reverse = True)

# select 10 topics
i = 0
for k,v in term_tuples:
    print k, v
    if i >= 10:
        break
    i+=1
    
print "done!"


###############################end of task2.py


##############################task3.py

`""" Classification


The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 


We would like to see a couple of steps:


1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time


Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?


"""

###################################################
# Solution 1 of Q3
# 1. Using textblob for a simple text classification
# 2. Build two classifiers: naive bayes and decision tree
# 3. Evaluation by a simplge cross validation for average accuracy
# All running experiments were run successfully on my own local mahine
###################################################

# import useful tools or package for different learning problems

from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import random

#from sklearn.datasets import fetch_20newsgroups
#from sklearn.datasets import load_files

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from numpy import genfromtxt, savetxt

# 1. load training data for building a simple naive bayes classifier for text classificaiton

bad_data = open('../data/bad_deals.txt','r')
dataset = []
for line in bad_data:
    dataset.append((line, 'bad'))

good_data = open('../data/good_deals.txt','r')
for line in good_data:
    dataset.append((line, 'good'))

test_data = open('../data/test_deals.txt','r')
test_dat = []
for line in test_data:
    test_dat.append((line))

# split for train and test data

results_nbc= []
results_dtc= []

# a simple cross valiadation for evaluation
for i in range(1, 10):

    # randomize
    random.seed(i)
    random.shuffle(dataset)

    # split
    train = dataset[:9*len(dataset)/10]
    test = dataset[9*len(dataset)/10:]

    # build classifiers
    nbc = NaiveBayesClassifier(train)
    dtc = DecisionTreeClassifier(train)

    # save results
    results_nbc.append(nbc.accuracy(test));
    results_dtc.append(dtc.accuracy(test));

# output the mean of accuray
print('mean of accuracy:')
print('naive bayes', np.array(results_nbc).mean())
print('decision tree', np.array(results_dtc).mean())

# 2. use test_deals.txt for classification

nbc = NaiveBayesClassifier(dataset)
dtc = DecisionTreeClassifier(dataset)
    
print('naive bayes classification:')
for text in test_dat:
    print(text, nbc.classify(text))

print('decision tree classification:')
for text in test_dat:
    print(text, dtc.classify(text))    
# other classifiers ...

#########################################enf of task3.py

###########################################task3b.py

""" Classification


The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 


We would like to see a couple of steps:


1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time


Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?


"""
############################################################################################
# Solution 2 for Q3
# 1. The solution is similar to the method for sentiment analysis, see below  
# 2. show how to split sentences and generate word features
# 3. raw data is transformed into a new training data, different from the method used in textblob;
# ref: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
#
# note: however, the performance is poorer than that by textblob, discussed in the first solution, from our empirical results
############################################################################################


# import useful tools or package for different learning problems

from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import random
import nltk
#from pylab import *

#from sklearn.datasets import fetch_20newsgroups
#from sklearn.datasets import load_files

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import DecisionTreeClassifier
from numpy import genfromtxt, savetxt
from sklearn.cross_validation import train_test_split

# 1. load training data for building a simple naive bayes classifier for text classificaiton
#

bad_data = open('../data/bad_deals.txt','r')
deals = []

good_data = open('../data/good_deals.txt','r')
for line in good_data:
    deals.append((line, 'good'))
    
for line in bad_data:
    deals.append((line, 'bad'))

test_data = open('../data/test_deals.txt','r')
test_dat = []
for line in test_data:
    test_dat.append((line))

# end

##names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
##         "Random Forest", "AdaBoost", "Naive Bayes"]
##classifiers = [
##    KNeighborsClassifier(3),
##    SVC(kernel="linear", C=0.025),
##    SVC(gamma=2, C=1),
##    DecisionTreeClassifier(max_depth = 5),
##    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
##    AdaBoostClassifier(),
##    NaiveBayesClassifier()]


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def get_words_in_deals(deals):
    all_words = []
    for (words, t) in deals:
        all_words.extend(words)
    return all_words

word_features = get_word_features(get_words_in_deals(deals))

def extract_features(deal):
    deal_words = set(deal)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in deal_words)
    return features   

training_set = nltk.classify.apply_features(extract_features, deals)

# evaluation by split
X_train = training_set[:int(len(training_set) *0.9)]
X_test = training_set[int(len(training_set) *0.9):]

clf= NaiveBayesClassifier(X_train)

# accuracy
print(clf.accuracy(X_test))

##target = [x[len(x)-1] for x in training_set]
##train = [x[:len(x)-2] for x in training_set]
##
##X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=.4)

##
### iterate over classifiers
##for name, clf in zip(names, classifiers):
##    clf.fit(X_train, y_train)
##
##    score = clf.score(X_test, y_test)
##    print(name, score)

# classify
classifier = nltk.NaiveBayesClassifier.train(training_set)
for deal in test_dat:
    print(classifier.classify(extract_features(deal.split())))


########################################end of task3b.py

