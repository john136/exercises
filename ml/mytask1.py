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

