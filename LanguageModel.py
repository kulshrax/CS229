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
		self.filename = filename
		self.corpus = PlaintextCorpusReader(corpusdir, filename)
		# Vector of vectors -> each sub-vector is a document 
		self.punctuation = set([',', ';', '\'', '"', '.', '!', '?'])
		self.allComments = self.splitBySpaces()
		self.words = [word for comment in self.allComments for word in comment]
		self.docCount = len(self.allComments)
		self.wordFreqs = Counter(self.words)
		self.bigramFreqs = Counter(ngrams(self.words, 2))
		self.trigramFreqs = Counter(ngrams(self.words, 3))
		
	def getDocCount(self):
		return self.docCount

	def getTotalWordCount(self):
		return sum([self.wordFreqs[a] for a in self.wordFreqs])

	def getWordFreqs(self):
		return self.wordFreqs

	def getSents(self):
		return self.corpus.sents()

	def getBigramFreqs(self):
		return self.bigramFreqs

	def getTotalBigramCount(self):
		return sum([self.bigramFreqs[a] for a in self.bigramFreqs])

	def getTrigramFreqs(self):
		return self.trigramFreqs

	def getTotalTrigramCount(self):
		return sum([self.trigramFreqs[a] for a in self.trigramFreqs])

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
