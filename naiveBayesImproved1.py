from LanguageModel import LanguageModel
from math import log

INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

def main():
	cleanLM = LanguageModel(CLEAN_TRAIN_FILE)
	insultLM = LanguageModel(INSULT_TRAIN_FILE)
	
	cleanTestSents = LanguageModel(CLEAN_TEST_FILE).getSents()
	insultTestSents = LanguageModel(INSULT_TEST_FILE).getSents()

	NB = baselineNaiveBayes(cleanLM, insultLM)
	NB.train()
	tp, tn, fp, fn = NB.test(cleanTestSents, insultTestSents)

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
		for word in self.cleanWordProbs:
			self.cleanWordProbs[word] = (self.cleanWordProbs[word] + 0.0) / self.cleanTotalWords
		self.insultWordProbs = self.insultLM.getWordFreqs()
		for word in self.insultWordProbs:
			self.insultWordProbs[word] = (self.insultWordProbs[word] + 0.0) / self.insultTotalWords

		# Calculate class priors
		self.cleanPrior = (self.numCleanSentences + 0.0) / (self.numCleanSentences + self.numInsultSentences)
		self.insultPrior = (self.numInsultSentences + 0.0) / (self.numCleanSentences + self.numInsultSentences)
		


if __name__ == "__main__":
	main()
