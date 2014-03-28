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
