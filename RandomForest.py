from sklearn.ensemble import RandomForestClassifier
from LanguageModel import LanguageModel
import numpy as np
from naiveBayesBaseline import baselineNaiveBayes
import tfidf
from sentiment import Sentiment
from Misspellings import Misspellings


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

    clf = RandomForestClassifier()
    print ("\tTraining random forest....")    
    clf.fit(trainMatrix, trainLabels)
    print ("\tTesting random forest....") 
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

    clf = RandomForestClassifier()
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


    tfidf_train_features = tfidf.make_feature_vectors(trainAACleanLM,
            trainAAInsultLM, trainABCleanLM, trainABInsultLM)

    tfidf_test_features = tfidf.make_feature_vectors(trainAACleanLM,
            trainAAInsultLM, testCleanLM, testInsultLM)

    print tfidf_test_features.shape, tfidf_train_features.shape
    print testMatrix.shape, trainMatrix.shape

    trainMatrix = np.hstack((trainMatrix, tfidf_train_features))
    testMatrix = np.hstack((testMatrix, tfidf_test_features))


    clf = RandomForestClassifier()
    print ("\tTraining random forest....")  
    clf.fit(trainMatrix, trainLabels)
    print ("\tTesting random forest....")   
    output3 = clf.predict(testMatrix).tolist()  

    ### SENTIMENT ###
    print("Running baseline + PoS Features + TF-IDF Features + Sentiment Features")
    s = Sentiment()
    clean_train = np.array(s.get_clean_train_vector())
    insult_train = np.array(s.get_insult_train_vector())
    sentiment_train_features = np.concatenate((clean_train, insult_train), axis=0)
    shape = sentiment_train_features.shape
    sentiment_train_features = sentiment_train_features.reshape((shape[0], 1))
    print sentiment_train_features.shape

    clean_test = np.array(s.get_clean_test_vector())
    insult_test = np.array(s.get_insult_test_vector())
    sentiment_test_features = np.concatenate((clean_test, insult_test), axis=0)
    shape = sentiment_test_features.shape
    sentiment_test_features = sentiment_test_features.reshape((shape[0], 1))
    print sentiment_test_features.shape

    trainMatrix = np.hstack((trainMatrix, sentiment_train_features))
    testMatrix = np.hstack((testMatrix, sentiment_test_features))

    clf = RandomForestClassifier()
    print ("\tTraining random forest....")  
    clf.fit(trainMatrix, trainLabels)
    print ("\tTesting random forest....")   
    output4 = clf.predict(testMatrix).tolist()  

    ### MISSPELLINGS ###
    print("Running baseline + PoS Features + TF-IDF Features + Sentiment Features + Misspellings features")
    m = Misspellings()
    clean_train = np.array(m.get_clean_misspellings(False))
    insult_train = np.array(m.get_insult_misspellings(False))
    misspellings_train_features = np.concatenate((clean_train, insult_train), axis=0)
    shape = misspellings_train_features.shape
    misspellings_train_features = misspellings_train_features.reshape((shape[0], 1))
    print misspellings_train_features.shape

    clean_test = np.array(m.get_clean_misspellings())
    insult_test = np.array(m.get_insult_misspellings())
    misspellings_test_features = np.concatenate((clean_test, insult_test), axis=0)
    shape = misspellings_test_features.shape
    misspellings_test_features = misspellings_test_features.reshape((shape[0], 1))
    print misspellings_test_features.shape

    trainMatrix = np.hstack((trainMatrix, sentiment_train_features))
    testMatrix = np.hstack((testMatrix, sentiment_test_features))

    clf = RandomForestClassifier()
    print ("\tTraining random forest....")  
    clf.fit(trainMatrix, trainLabels)
    print ("\tTesting forest....")   
    output5 = clf.predict(testMatrix).tolist()  

    with open('RANDOM_FOREST_output_file_without_SB.txt', 'w+') as f:
        f.write("Output 1\n")
        f.write("{}\n".format(output1))
        interpret_results(output1, testLabels, f)
        f.write("\nOutput 2\n") 
        f.write("{}\n".format(output2))
        interpret_results(output2, testLabels, f)
        f.write("\nOutput 3\n") 
        f.write("{}\n".format(output3))
        interpret_results(output3, testLabels, f)
        f.write("Output 4\n")
        f.write("{}\n".format(output4))
        interpret_results(output4, testLabels, f)
        f.write("Output 5\n")
        f.write("{}\n".format(output5))
        interpret_results(output5, testLabels, f)


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