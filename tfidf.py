#!/usr/bin/env python

from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from naiveBayesBaseline import interpretResults
from LanguageModel import LanguageModel
import numpy as np
from scipy.spatial.distance import cosine

CLEAN_TRAIN_CORPUS = 'corpora/clean_corpus_train.txt'
INSULT_TRAIN_CORPUS = 'corpora/insult_corpus_train.txt'

CLEAN_TEST_CORPUS = 'corpora/clean_corpus_test.txt'
INSULT_TEST_CORPUS = 'corpora/insult_corpus_test.txt'

CLEAN_TRAIN_AA_FILE = 'clean_corpus_train1aa.txt'
CLEAN_TRAIN_AB_FILE = 'clean_corpus_train1ab.txt'

INSULT_TRAIN_AA_FILE = 'insult_corpus_train1aa.txt'
INSULT_TRAIN_AB_FILE = 'insult_corpus_train1ab.txt'


def train(train_documents):
    """Convert a list of strings into a matrix where each row represents a 
       string and each column represents the TF-IDF normalized count of each
       word incountered in that document. Also returns the vectorizer so that
       we can convert other documents into the same vector format for
       classification.
    """
    vectorizer = TfidfVectorizer()
    counts = vectorizer.fit_transform(train_documents)
    return vectorizer, counts


def classify(vectorizer, counts, test_documents):
    """Takes in a vectorizer and count matrix from train() and uses them
       to classify a list of documents. Returns a tuple for each document
       containing the predicted class and the scores against each of the
       training documents.
    """
    document_matrix = vectorizer.transform(test_documents)
    results = []
    similarity = lambda a, b: 1 - cosine(a.toarray()[0], b.toarray()[0])
    for document in document_matrix:
        scores = [similarity(document, row) for row in counts]
        predicted_class = scores.index(max(scores))
        results.append((predicted_class, scores))
    return results


def make_feature_vectors(train_clean_lm, train_insult_lm,
        feature_clean_lm, feature_insult_lm):
    train_clean = train_clean_lm.getRawText()
    train_insult = train_insult_lm.getRawText()
    train_corpus = [train_clean, train_insult]

    vec, counts = train(train_corpus)

    test_clean = feature_clean_lm.getLines()
    test_insults = feature_insult_lm.getLines()
    test_docs = test_clean + test_insults

    return np.array([scores for _, scores in classify(vec, counts, test_docs)])

def test():
    trainAACleanLM = LanguageModel(CLEAN_TRAIN_AA_FILE)
    trainAAInsultLM = LanguageModel(INSULT_TRAIN_AA_FILE)
        
    trainABCleanLM = LanguageModel(CLEAN_TRAIN_AB_FILE)
    trainABInsultLM = LanguageModel(INSULT_TRAIN_AB_FILE)

    return make_training_vectors(trainAACleanLM, trainAAInsultLM,
        trainABCleanLM, trainABInsultLM)

def test_classifier():
    clean_train_doc = open(CLEAN_TRAIN_CORPUS).read()
    insult_train_doc = open(INSULT_TRAIN_CORPUS).read()
    train_corpus = [clean_train_doc, insult_train_doc]

    vec, counts = train(train_corpus)

    clean_test_docs = list(open(CLEAN_TEST_CORPUS))
    insult_test_docs = list(open(INSULT_TEST_CORPUS))

    clean_results = classify(vec, counts, clean_test_docs)
    insult_results = classify(vec, counts, insult_test_docs)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for predicted_class, _ in clean_results:
        if predicted_class == 0:
            true_negatives += 1
        else:
            false_positives += 1

    for predicted_class, _ in insult_results:
        if predicted_class == 0:
            false_negatives += 1
        else:
            true_positives += 1

    total = true_negatives + true_positives + false_negatives + false_negatives
    correct_percent = (true_positives + true_negatives) / total

    print correct_percent
    interpretResults(true_positives, true_negatives, false_positives, false_negatives)
    return true_positives, true_negatives, false_positives, false_negatives


def main(argv):
    test_classifier()


if __name__ == '__main__':
    main(sys.argv)