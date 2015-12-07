from sklearn import svm
from LanguageModel import LanguageModel
import numpy as np
from naiveBayesBaseline import baselineNaiveBayes


INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TRAIN_SHORT_FILE = 'insult_corpus_train_short.txt'
CLEAN_TRAIN_SHORT_FILE = 'clean_corpus_train_short.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

INSULT_TEST_SHORT_FILE = 'insult_corpus_test_short.txt'
CLEAN_TEST_SHORT_FILE = 'clean_corpus_test_short.txt'

CLEAN_TRAIN_AA_FILE = 'clean_corpus_train1aa.txt'
CLEAN_TRAIN_AB_FILE = 'clean_corpus_train1ab.txt'

INSULT_TRAIN_AA_FILE = 'insult_corpus_train1aa.txt'
INSULT_TRAIN_AB_FILE = 'insult_corpus_train1ab.txt'

def main():
	print ("Generating language models....")
	trainAACleanLM = LanguageModel(CLEAN_TRAIN_AA_FILE)
	trainAAInsultLM = LanguageModel(INSULT_TRAIN_AA_FILE)
		
	trainABCleanLM = LanguageModel(CLEAN_TRAIN_AB_FILE)
	trainABInsultLM = LanguageModel(INSULT_TRAIN_AB_FILE)

	testCleanLM = LanguageModel(CLEAN_TEST_FILE)
	testInsultLM = LanguageModel(INSULT_TEST_FILE)

	trainLabels = np.array(([0] * trainABCleanLM.getDocCount()) + ([1] * trainABInsultLM.getDocCount()))
	testLabels = np.array(([0] * testCleanLM.getDocCount()) + ([1] * testInsultLM.getDocCount()))


	### Just baseline probabilities
	print ("Running baseline....")
	NB = baselineNaiveBayes(trainAACleanLM, trainAAInsultLM)
	print ("\tTraining NB....")	
	NB.train()
	print ("\tTesting NB....")	
	totalNBMatrix = np.array(NB.genProbs(trainABCleanLM.getSents(), trainABInsultLM.getSents()))

	trainMatrix = totalNBMatrix 

	testMatrix = np.array(NB.genProbs(testCleanLM.getSents(), testInsultLM.getSents()))

	clf = svm.SVC()
	print ("\tTraining SVM....")	
	clf.fit(trainMatrix, trainLabels)
	print ("\tTesting SVM....")	
	output1 = clf.predict(testMatrix).tolist()


	### Baseline + PoS Features
	print ("Running baseline + PoS Features....")
	cleanPosMatrix = trainABCleanLM.getPosMatrix()
	insultPosMatrix = trainABInsultLM.getPosMatrix()

	testCleanPosMatrix = testCleanLM.getPosMatrix()
	testInsultPosMatrix = testInsultLM.getPosMatrix()

	posFeatures = np.array(cleanPosMatrix + insultPosMatrix)
	testPosFeatures = np.array(testCleanPosMatrix + testInsultPosMatrix)
	trainMatrix = np.hstack((trainMatrix, posFeatures))
	testMatrix = np.hstack((testMatrix, testPosFeatures))

	clf = svm.SVC()
	print ("\tTraining SVM....")	
	clf.fit(trainMatrix, trainLabels)
	print ("\tTesting SVM....")	
	output2 = clf.predict(testMatrix).tolist()


	### Baseline + PoS Features + TF-IDF Features (TODO Arun)
	print("Running baseline + PoS Features + TF-IDF Features")
	# generate list of features with TFIDF, using trainABCleanLM and trainABInsultLM
	# trainMatrix = np.hstack((trainMatrix, the new thing you just generated))
	# do same for testMatrix
	# clf = svm.SVC()
	# print ("\tTraining SVM....")	
	# clf.fit(trainMatrix, trainLabels)
	# print ("\tTesting SVM....")	
	# output3 = clf.predict(testMatrix).tolist()	
	# then update the output_file.txt thing below



	with open('output_file.txt', 'w+') as f:
		f.write("Output 1\n")
		f.write("{}\n".format(output1))
		interpret_results(output1, testLabels, f)
		f.write("\nOutput 2\n")	
		f.write("{}\n".format(output2))
		interpret_results(output2, testLabels, f)


def interpret_results(output, testLabels, f):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i in xrange(len(output)):
		if testLabels[i] == 1:
			if output[i] == 1:
				tp += 1
			else:
				fn += 1
		else:
			if output[i] == 1:
				fp += 1
			else:
				tn += 1

	if (tp + fp) == 0:
		precision = 0.0
	else:
		precision = (tp + 0.0) / (tp + fp)
	if (tp + fn) == 0:
		recall = 0.0
	else:
		recall = (tp + 0.0) / (tp + fn)
	
	if (precision + recall) == 0.0:
		f1 = 0.0
	else:
		f1 = 2 * (precision * recall) / (precision + recall)

	f.write("~~~~~~~ Results ~~~~~~~\n")
	f.write("Precision: %.3f\n" % precision)
	f.write("Recall: %.3f\n" % recall)
	f.write("F1 Score: %.3f\n " % f1)
	f.write("tp: {}, tn: {}, fp: {}, fn: {}\n".format(tp, tn, fp, fn))


	return tp, tn, fp, fn, precision, recall, f1


if __name__ == "__main__":
	main()