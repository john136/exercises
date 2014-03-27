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
