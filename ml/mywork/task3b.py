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

#
# 1. The solution is similar to the method for sentiment analysis, see below  
# 2. show how to split sentences and generate word features
# 3. raw data is transformed into a new training data, different from the method used in textblob;
# ref: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
#
# note: however, the performance is poorer than that by textblob, from empirical results

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
