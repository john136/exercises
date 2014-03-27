""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

# solution:
# using termextract and tagging tools for term extraction; 
# ranking for top k popular terms 
# results for answering 1 and 2
#

from topia.termextract import extract
from topia.termextract import tag

tagger = tag.Tagger()
tagger.initialize()

extractor = extract.TermExtractor()
extractor.filter = extract.permissiveFilter

# load data
openfile = open('..\data\deals.txt', 'r')

# dictionary
d = {}

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
