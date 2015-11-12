import os
from collections import Counter
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

corpusdir = 'corpora/' # Directory of corpus.

#cleanCorpus = PlaintextCorpusReader(corpusdir, 'clean_corpus_train.txt')
#insultCorpus = PlaintextCorpusReader(corpusdir, 'insult_corpus_train.txt')

#print Counter(cleanCorpus.words())

class LanguageModel:
	def __init__(self, filename):
		self.corpus = PlaintextCorpusReader(corpusdir, filename)
		self.docCount = len(self.corpus.sents())
		self.wordFreqs = Counter(self.corpus.words())
		
	def getDocCount(self):
		return self.docCount

	def getWordFreqs(self):
		return self.wordFreqs

	def getSents(self):
		return self.corpus.sents()

	def getRawDump(self):
		return self.corpus.raw().strip()
