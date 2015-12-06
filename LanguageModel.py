import os
from collections import Counter
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.util import ngrams
import pos

corpusdir = 'corpora/' # Directory of corpus.

#cleanCorpus = PlaintextCorpusReader(corpusdir, 'clean_corpus_train.txt')
#insultCorpus = PlaintextCorpusReader(corpusdir, 'insult_corpus_train.txt')

#print Counter(cleanCorpus.words())

class LanguageModel:
	def __init__(self, filename):
		self.corpus = PlaintextCorpusReader(corpusdir, filename)
		self.docCount = len(self.corpus.sents())
		self.wordFreqs = Counter(self.corpus.words())
		self.bigramFreqs = Counter(ngrams(self.corpus.words(), 2))
		self.trigramFreqs = Counter(ngrams(self.corpus.words(), 3))
		self.posFracs = pos.pos_fractions(self.corpus.words())
		
	def getDocCount(self):
		return self.docCount

	def getTotalWordCount(self):
		return sum([self.wordFreqs[a] for a in self.wordFreqs])

	def getWordFreqs(self):
		return self.wordFreqs

	def getSents(self):
		return self.corpus.sents()

	def getRawDump(self):
		return self.corpus.raw().strip()

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
