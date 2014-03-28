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

