import os
from collections import Counter
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.util import ngrams
import pos

corpusdir = 'corpora/' # Directory of corpus.

INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

#cleanCorpus = PlaintextCorpusReader(corpusdir, 'clean_corpus_train.txt')
#insultCorpus = PlaintextCorpusReader(corpusdir, 'insult_corpus_train.txt')

#print Counter(cleanCorpus.words())

class LanguageModel:
    def __init__(self, filename):
        self.filename = filename
        self.raw = open(corpusdir+filename).read()
        self.lines = list(open(corpusdir+filename))
        # Vector of vectors -> each sub-vector is a document 
        self.punctuation = set([',', ';', '\'', '"', '.', '!', '?'])
        self.allComments = self.splitBySpaces()
        self.words = [word for comment in self.allComments for word in comment]
        self.docCount = len(self.allComments)
        self.wordFreqs = Counter(self.words)
        self.bigramFreqs = Counter(ngrams(self.words, 2))
        self.trigramFreqs = Counter(ngrams(self.words, 3))
        self.posFracs = pos.pos_fractions(self.words)
        
    def getRawText(self):
        return self.raw

    def getLines(self):
        return self.lines

    def getDocCount(self):
        return self.docCount

    def getTotalWordCount(self):
        return sum([self.wordFreqs[a] for a in self.wordFreqs])

    def getWordFreqs(self):
        return self.wordFreqs

    def getSents(self):
        return self.allComments

    def getBigramFreqs(self):
        return self.bigramFreqs

    def getTotalBigramCount(self):
        return sum([self.bigramFreqs[a] for a in self.bigramFreqs])

    def getTrigramFreqs(self):
        return self.trigramFreqs

    def getTotalTrigramCount(self):
        return sum([self.trigramFreqs[a] for a in self.trigramFreqs])

    def getNounFrac(self):
        return self.posFracs['N']

    def getVerbFrac(self):
        return self.posFracs['V']

    def getAdjFrac(self):
       return self.posFracs['A']

    def getPosMatrix(self):
        posFracs = (pos.pos_fractions(c) for c in self.allComments)
        return [[i['N'], i['V'], i['A']] for i in posFracs]


    def splitBySpaces(self):
        open_file = open(corpusdir+self.filename)
        to_return = []
        for line in open_file:
            for char in self.punctuation:
                line = line.replace(char, '')
            tokens = line.split()
            to_return.append(tokens)
        open_file.close()
        return to_return

def trainCleanLM():
    return LanguageModel(CLEAN_TRAIN_FILE)

def trainInsultLM(): 
    return LanguageModel(INSULT_TRAIN_FILE)
    
def testCleanLM():
    return LanguageModel(CLEAN_TEST_FILE)

def testInsultLM():
    return LanguageModel(INSULT_TEST_FILE)