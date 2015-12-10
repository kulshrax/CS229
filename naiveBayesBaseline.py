from LanguageModel import LanguageModel
from math import log
from nltk.corpus import stopwords
from collections import Counter

INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

CLEAN_TRAIN_AA_FILE = 'clean_corpus_train1aa.txt'
CLEAN_TRAIN_AB_FILE = 'clean_corpus_train1ab.txt'

INSULT_TRAIN_AA_FILE = 'insult_corpus_train1aa.txt'
INSULT_TRAIN_AB_FILE = 'insult_corpus_train1ab.txt'

LAPLACE_SMOOTHING = True
LAPLACE_SMOOTHER = 0.01

REMOVE_STOPWORDS = False
STUPID_BACKOFF = True
USING_TRIGRAM = True
SB_ALPHA = 0.01 #discount factor for stupid backoff
ALPHA = 1.0

def main():

	precisions = []
	recalls = []
	for alpha in [0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 1.00, 1.01, 1.03, 1.05, 1.07, 1.1, 1.3, 1.5, 1.7, 2.0, 3.0, 5.0]:
		ALPHA = alpha

		cleanLM = LanguageModel(CLEAN_TRAIN_FILE)
		insultLM = LanguageModel(INSULT_TRAIN_FILE)
		
		cleanTestSents = LanguageModel(CLEAN_TEST_FILE).getSents()
		insultTestSents = LanguageModel(INSULT_TEST_FILE).getSents()

		NB = baselineNaiveBayes(cleanLM, insultLM)
		NB.train()
		#print NB.genProbs(cleanTestSents, insultTestSents)

		if (STUPID_BACKOFF):
			tp, tn, fp, fn = NB.testStupidBackoff(cleanTestSents, insultTestSents, ALPHA)
		else:
			tp, tn, fp, fn = NB.testImproved1(cleanTestSents, insultTestSents, ALPHA)

		interpretResults(tp, tn, fp, fn)

	print "Precisions:\n {}".format(precisions)
	print "Recalls:\n {}".format(recalls)

def interpretResults(tp, tn, fp, fn):
	precision = (tp + 0.0) / (tp + fp)
	recall = (tp + 0.0) / (tp + fn)
	f1 = 2 * (precision * recall) / (precision + recall)

	print "~~~~~~~ Results ~~~~~~~"
	print "Precision: %.3f" % precision
	print "Recall: %.3f" % recall
	print "F1 Score: %.3f " % f1
	print "tp: {}, tn: {}, fp: {}, fn: {}".format(tp, tn, fp, fn)

class baselineNaiveBayes:
	# does NOTHING intelligent, is purposefully simple
	
	def __init__(self, cleanLM, insultLM):
		self.cleanLM = cleanLM
		self.insultLM = insultLM

		self.cleanTotalWords = cleanLM.getTotalWordCount()
		self.insultTotalWords = insultLM.getTotalWordCount()
		self.cleanWordFreqs = cleanLM.getWordFreqs()
		self.insultWordFreqs = insultLM.getWordFreqs()
		self.numCleanSentences = cleanLM.getDocCount()
		self.numInsultSentences = insultLM.getDocCount()

		# set by train()
		self.cleanWordProbs = None
		self.insultWordProbs = None
		self.cleanPrior = None
		self.insultPrior = None

		# set by train() for stupid backoff
		self.cleanBigramProbs = Counter()
		self.insultBigramProbs = Counter()
		self.cleanTrigramProbs = Counter()
		self.insultTrigramProbs = Counter()

	def train(self):
		# Calculate word probabilities

		self.cleanWordProbs = self.cleanLM.getWordFreqs()
		self.insultWordProbs = self.insultLM.getWordFreqs()

		if (REMOVE_STOPWORDS):
			for stopword in stopwords.words('english'):
				self.cleanWordProbs[stopword] = 0.0
				self.insultWordProbs[stopword] = 0.0
		
		if (LAPLACE_SMOOTHING):
			for word in (self.cleanWordProbs + self.insultWordProbs):
				self.cleanWordProbs[word] = (self.cleanWordProbs[word] + LAPLACE_SMOOTHER) / self.cleanTotalWords
				self.insultWordProbs[word] = (self.insultWordProbs[word] + LAPLACE_SMOOTHER) / self.insultTotalWords

		else:
			for word in self.cleanWordProbs:
				self.cleanWordProbs[word] = (self.cleanWordProbs[word] + 0.0) / self.cleanTotalWords
			for word in self.insultWordProbs:
				self.insultWordProbs[word] = (self.insultWordProbs[word] + 0.0) / self.insultTotalWords

		if (STUPID_BACKOFF):
			cleanBigramFreqs = self.cleanLM.getBigramFreqs()
			insultBigramFreqs = self.insultLM.getBigramFreqs()
			cleanBigramTotal = self.cleanLM.getTotalBigramCount()
			insultBigramTotal = self.insultLM.getTotalBigramCount()

			cleanTrigramFreqs = self.cleanLM.getTrigramFreqs()
			insultTrigramFreqs = self.insultLM.getTrigramFreqs()
			cleanTrigramTotal = self.cleanLM.getTotalTrigramCount()
			insultTrigramTotal = self.insultLM.getTotalTrigramCount()

			for word in cleanBigramFreqs:
				self.cleanBigramProbs[word] = (cleanBigramFreqs[word] + 0.0) / cleanBigramTotal
			for word in insultBigramFreqs:
				self.insultBigramProbs[word] = (insultBigramFreqs[word] + 0.0) / insultBigramTotal

			for word in cleanTrigramFreqs:
				self.cleanTrigramProbs[word] = (cleanTrigramFreqs[word] + 0.0) / cleanTrigramTotal
			for word in insultBigramFreqs:
				self.insultTrigramProbs[word] = (insultTrigramFreqs[word] + 0.0) / insultTrigramTotal


		# Calculate class priors
		self.cleanPrior = (self.numCleanSentences + 0.0) / (self.numCleanSentences + self.numInsultSentences)
		self.insultPrior = (self.numInsultSentences + 0.0) / (self.numCleanSentences + self.numInsultSentences)
		

	def test(self, cleanSents, insultSents):
		truePos = 0 # Correctly-labeled insults
		trueNeg = 0 # Correctly-labeled clean
		falsePos = 0 # Clean mislabeled as insult
		falseNeg = 0 # Insult mislabeled as clean
		
		#print self.cleanWordProbs
		#print self.insultWordProbs
				
		for sentence in cleanSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for word in sentence:
				#print "Word {}, cleanProb {}, insultProb {}".format(word, self.cleanWordProbs[word], self.insultWordProbs[word])
				if (self.cleanWordProbs[word] > 0):
					cleanProb += log(self.cleanWordProbs[word])
				else:
					cleanProb = float("-inf")
				if (self.insultWordProbs[word] > 0):
					insultProb += log(self.insultWordProbs[word])
				else:
					insultProb = float("-inf")
			#print "CleanProb {}, InsultProb {}".format(cleanProb, insultProb)
			if (cleanProb > insultProb):
				truePos += 1
			else:
				falseNeg += 1


		for sentence in insultSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for word in sentence:
				if (self.cleanWordProbs[word] > 0):
					cleanProb += log(self.cleanWordProbs[word])
				else:
					cleanProb = float("-inf")
				if (self.insultWordProbs[word] > 0):
					insultProb += log(self.insultWordProbs[word])
				else:
					insultProb = float("-inf")
			if (cleanProb > insultProb):
				falsePos += 1
			else:
				trueNeg += 1

		return truePos, trueNeg, falsePos, falseNeg

	def genProbs(self, cleanSents, insultSents):
		probs = []
		for sentence in cleanSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for i in xrange(len(sentence)):
				if (i < len(sentence) - 2):
					trigram = (sentence[i], sentence[i+1], sentence[i+2])
				else:
					trigram = None
				if (i < len(sentence) - 1):
					bigram = (sentence[i], sentence[i+1]) 
				else:
					bigram = None
				unigram = sentence[i]
				bigramCleanProb = self.cleanBigramProbs[bigram]
				bigramInsultProb = self.insultBigramProbs[bigram]
				trigramCleanProb = self.cleanTrigramProbs[trigram]
				trigramInsultProb = self.insultTrigramProbs[trigram]

				# Use clean bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					cleanProb += log(trigramCleanProb)
				elif bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					cleanProb += log(SB_ALPHA * bigramCleanProb)
				elif (self.cleanWordProbs[unigram] > 0 and self.insultWordProbs[unigram] > 0):
					cleanProb += log(SB_ALPHA * SB_ALPHA * self.cleanWordProbs[unigram])

				# Use insult bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					insultProb += log(trigramInsultProb)
				elif bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					insultProb += log(SB_ALPHA * bigramInsultProb)
				elif (self.cleanWordProbs[unigram] > 0 and self.insultWordProbs[unigram] > 0):
					insultProb += log(SB_ALPHA * SB_ALPHA * self.insultWordProbs[unigram])
				
			probs.append([cleanProb, insultProb])

		for sentence in insultSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for i in xrange(len(sentence)-2):
				if (i < len(sentence) - 2):
					trigram = (sentence[i], sentence[i+1], sentence[i+2])
				else:
					trigram = None
				if (i < len(sentence) - 1):
					bigram = (sentence[i], sentence[i+1]) 
				else:
					bigram = None 
				bigramCleanProb = self.cleanBigramProbs[bigram]
				bigramInsultProb = self.insultBigramProbs[bigram]
				trigramCleanProb = self.cleanTrigramProbs[trigram]
				trigramInsultProb = self.insultTrigramProbs[trigram]

				# Use clean bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					cleanProb += log(trigramCleanProb)
				elif bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					cleanProb += log(SB_ALPHA * bigramCleanProb)
				elif (self.cleanWordProbs[unigram] > 0 and self.insultWordProbs[unigram] > 0):
					cleanProb += log(SB_ALPHA * SB_ALPHA * self.cleanWordProbs[unigram])


				# Use insult bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					insultProb += log(trigramInsultProb)
				elif bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					insultProb += log(SB_ALPHA * bigramInsultProb)
				elif (self.cleanWordProbs[unigram] > 0 and self.insultWordProbs[unigram] > 0):
					insultProb += log(SB_ALPHA * SB_ALPHA * self.insultWordProbs[unigram])

			probs.append([cleanProb, insultProb])				

		return probs


	# This version tried simply ignoring words that don't appear in either LM.
	def testImproved1(self, cleanSents, insultSents, ALPHA):
		truePos = 0 # Correctly-labeled insults
		trueNeg = 0 # Correctly-labeled clean
		falsePos = 0 # Clean mislabeled as insult
		falseNeg = 0 # Insult mislabeled as clean
		
		#print self.cleanWordProbs
		#print self.insultWordProbs
		
		
		
		for sentence in cleanSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for word in sentence:
				#print "cleanProb: {}, insultProb: {}".format(self.cleanWordProbs[word], self.insultWordProbs[word])
				if (self.cleanWordProbs[word] > 0 and self.insultWordProbs[word] > 0):
					cleanProb += log(self.cleanWordProbs[word])
					insultProb += log(self.insultWordProbs[word])
			#print "cleanProb {}, insultProb {}".format(cleanProb, insultProb)
			if ((cleanProb + 0.0) / insultProb <= ALPHA):
				truePos += 1
			else:
				falseNeg += 1


		for sentence in insultSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for word in sentence:
				if (self.cleanWordProbs[word] > 0 and self.insultWordProbs[word] > 0):
					cleanProb += log(self.cleanWordProbs[word])
					insultProb += log(self.insultWordProbs[word])

			if ((cleanProb + 0.0) / insultProb <= ALPHA):
				falsePos += 1
			else:
				trueNeg += 1

		return truePos, trueNeg, falsePos, falseNeg

	def testStupidBackoff(self, cleanSents, insultSents, ALPHA):
		truePos = 0 # Correctly-labeled insults
		trueNeg = 0 # Correctly-labeled clean
		falsePos = 0 # Clean mislabeled as insult
		falseNeg = 0 # Insult mislabeled as clean
		
		# Based off code I wrote for a CS124 assignment 

		for sentence in cleanSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for i in xrange(len(sentence)):
				if (i < len(sentence) - 2):
					trigram = (sentence[i], sentence[i+1], sentence[i+2])
				else:
					trigram = None
				if (i < len(sentence) - 1):
					bigram = (sentence[i], sentence[i+1]) 
				else:
					bigram = None
				unigram = sentence[i]
				bigramCleanProb = self.cleanBigramProbs[bigram]
				bigramInsultProb = self.insultBigramProbs[bigram]
				trigramCleanProb = self.cleanTrigramProbs[trigram]
				trigramInsultProb = self.insultTrigramProbs[trigram]

				# Use clean bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					cleanProb += log(trigramCleanProb)
				if bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					cleanProb += log(SB_ALPHA * bigramCleanProb)
				elif (self.cleanWordProbs[unigram] > 0 and self.insultWordProbs[unigram] > 0):
					cleanProb += log(SB_ALPHA * SB_ALPHA * self.cleanWordProbs[unigram])

				# Use insult bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					insultProb += log(trigramInsultProb)
				if bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					insultProb += log(SB_ALPHA * bigramInsultProb)
				elif (self.cleanWordProbs[unigram] > 0 and self.insultWordProbs[unigram] > 0):
					insultProb += log(SB_ALPHA * SB_ALPHA * self.insultWordProbs[unigram])
				
			if ((cleanProb + 0.0) / insultProb <= ALPHA):
				truePos += 1
			else:
				falseNeg += 1

		for sentence in insultSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for i in xrange(len(sentence)-2):
				if (i < len(sentence) - 2):
					trigram = (sentence[i], sentence[i+1], sentence[i+2])
				else:
					trigram = None
				if (i < len(sentence) - 1):
					bigram = (sentence[i], sentence[i+1]) 
				else:
					bigram = None 
				bigramCleanProb = self.cleanBigramProbs[bigram]
				bigramInsultProb = self.insultBigramProbs[bigram]
				trigramCleanProb = self.cleanTrigramProbs[trigram]
				trigramInsultProb = self.insultTrigramProbs[trigram]

				# Use clean bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					cleanProb += log(trigramCleanProb)
				if bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					cleanProb += log(SB_ALPHA * bigramCleanProb)

				# Use insult bigram else unigram
				if USING_TRIGRAM and trigramCleanProb > 0.0 and trigramInsultProb > 0.0:
					insultProb += log(trigramInsultProb)
				if bigramCleanProb > 0.0 and bigramInsultProb > 0.0:
					insultProb += log(SB_ALPHA * bigramInsultProb)

			if ((cleanProb + 0.0) / insultProb <= ALPHA):
				falsePos += 1
			else:
				trueNeg += 1


		return truePos, trueNeg, falsePos, falseNeg


if __name__ == "__main__":
	main()
