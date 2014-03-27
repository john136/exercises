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

# import useful tools or package for different learning problems

from __future__ import print_function
import logging
#import pandas as pd
import numpy as np
from optparse import OptionParser
import sys
from time import time

#import pylab as pl

#sys.path.append('..')

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files

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
from numpy import genfromtxt, savetxt


# 1. load training data for building a simple naive bayes classifier for text classificaiton

bad_data = open('../data/bad_deals.txt','r')
train = []
for line in bad_data:
    train.append((line, 'bad'))

good_data = open('../data/good_deals.txt','r')
for line in good_data:
    train.append((line, 'good'))

test_data = open('../data/test_deals.txt','r')
test = []
for line in test_data:
    test.append((line))

cl = NaiveBayesClassifier(train)

# 2. use test_deals.txt for classification
for text in test:
    print(text, cl.classify(text))
    
# other classifiers ...
