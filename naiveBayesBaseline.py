from LanguageModel import LanguageModel
from math import log
from nltk.corpus import stopwords

INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

LAPLACE_SMOOTHING = True
LAPLACE_SMOOTHER = 0.1

REMOVE_STOPWORDS = True

def main():
	cleanLM = LanguageModel(CLEAN_TRAIN_FILE)
	insultLM = LanguageModel(INSULT_TRAIN_FILE)
	
	cleanTestSents = LanguageModel(CLEAN_TEST_FILE).getSents()
	insultTestSents = LanguageModel(INSULT_TEST_FILE).getSents()

	NB = baselineNaiveBayes(cleanLM, insultLM)
	NB.train()
	tp, tn, fp, fn = NB.testImproved1(cleanTestSents, insultTestSents)

	interpretResults(tp, tn, fp, fn)

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

		# Calculate class priors
		self.cleanPrior = (self.numCleanSentences + 0.0) / (self.numCleanSentences + self.numInsultSentences)
		self.insultPrior = (self.numInsultSentences + 0.0) / (self.numCleanSentences + self.numInsultSentences)
		

	def test(self, cleanSents, insultSents):
		truePos = 0 # Correctly-labeled insults
		trueNeg = 0 # Correctly-labeled clean
		falsePos = 0 # Clean mislabeled as insult
		falseNeg = 0 # Insult mislabeled as clean
		
		print self.cleanWordProbs
		print self.insultWordProbs
				
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
			print "CleanProb {}, InsultProb {}".format(cleanProb, insultProb)
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

	# This version tried simply ignoring words that don't appear in either LM.
	def testImproved1(self, cleanSents, insultSents):
		truePos = 0 # Correctly-labeled insults
		trueNeg = 0 # Correctly-labeled clean
		falsePos = 0 # Clean mislabeled as insult
		falseNeg = 0 # Insult mislabeled as clean
		
		print self.cleanWordProbs
		print self.insultWordProbs
		
		
		
		for sentence in cleanSents:
			cleanProb = log(self.cleanPrior)
			insultProb = log(self.insultPrior)
			for word in sentence:
				print "cleanProb: {}, insultProb: {}".format(self.cleanWordProbs[word], self.insultWordProbs[word])
				if (self.cleanWordProbs[word] > 0 and self.insultWordProbs[word] > 0):
					cleanProb += log(self.cleanWordProbs[word])
					insultProb += log(self.insultWordProbs[word])
			print "cleanProb {}, insultProb {}".format(cleanProb, insultProb)
			if (cleanProb > insultProb):
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

			if (cleanProb > insultProb):
				falsePos += 1
			else:
				trueNeg += 1

		return truePos, trueNeg, falsePos, falseNeg


if __name__ == "__main__":
	main()
