from sklearn import svm
from LanguageModel import LanguageModel
import numpy as np


INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TRAIN_SHORT_FILE = 'insult_corpus_train_short.txt'
CLEAN_TRAIN_SHORT_FILE = 'clean_corpus_train_short.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

INSULT_TEST_SHORT_FILE = 'insult_corpus_test_short.txt'
CLEAN_TEST_SHORT_FILE = 'clean_corpus_test_short.txt'

def main():
	trainCleanLM = LanguageModel(CLEAN_TRAIN_SHORT_FILE)
	trainInsultLM = LanguageModel(INSULT_TRAIN_SHORT_FILE)
		
	testCleanLM = LanguageModel(CLEAN_TEST_SHORT_FILE)
	testInsultLM = LanguageModel(INSULT_TEST_SHORT_FILE)

	cleanPosMatrix = trainCleanLM.getPosMatrix()
	insultPosMatrix = trainInsultLM.getPosMatrix()

	trainMatrix = np.array(cleanPosMatrix + insultPosMatrix)
	trainLabels = np.array(([0] * len(cleanPosMatrix)) + ([1] * len(insultPosMatrix)))

	print "Generated models"
	clf = svm.SVC()
	print "Fitting data...."
	clf.fit(trainMatrix, trainLabels)
	print "foo"

	testCleanPosMatrix = testCleanLM.getPosMatrix()
	testInsultPosMatrix = testInsultLM.getPosMatrix()
	
	testMatrix = np.array(testCleanPosMatrix + testInsultPosMatrix)

	print clf.predict(testMatrix)














if __name__ == "__main__":
	main()