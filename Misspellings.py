#return vector of fraction of words misspelled for each of the training examples

import enchant
from LanguageModel import LanguageModel
import numpy as np

class Misspellings(object):

  def __init__(self):

    INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
    CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

    INSULT_TEST_FILE = 'insult_corpus_test.txt'
    CLEAN_TEST_FILE = 'clean_corpus_test.txt'

    self.punctuation = set([',', ';', '\'', '"', '.', '!', '?'])
    self.dictionary = enchant.Dict("en_US")

    self.cleanTrainSents = LanguageModel(CLEAN_TRAIN_FILE).getSents()
    self.insultTrainSents = LanguageModel(INSULT_TRAIN_FILE).getSents()

    self.cleanTestSents = LanguageModel(CLEAN_TEST_FILE).getSents()
    self.insultTestSents = LanguageModel(INSULT_TEST_FILE).getSents()

    self.cleanSplitSpaces = LanguageModel(CLEAN_TRAIN_FILE).splitBySpaces()
    self.insultSplitSpaces = LanguageModel(INSULT_TRAIN_FILE).splitBySpaces()

    self.cleanTestSplitSpaces = LanguageModel(CLEAN_TEST_FILE).splitBySpaces()
    self.insultTestSplitSpaces = LanguageModel(INSULT_TEST_FILE).splitBySpaces()

  def get_clean_misspellings(self, isTest=True):
    if isTest:
      fileName = self.cleanTestSplitSpaces
    else:
      fileName = self.cleanSplitSpaces
    vector = []
    for sent in fileName:
      misspelled_words = 0
      total_words = 0
      for word in sent:
        word = word.strip()
        #skip numbers, punctuations, non-words(i guess this is my splitting?), and usernames
        if word.isdigit() or word in self.punctuation or len(word) == 0 or word[0] == '@':
          continue
        total_words += 1
        if not self.dictionary.check(word):
          misspelled_words += 1
      if total_words == 0:
        continue
      vector.append(misspelled_words/float(total_words))
    return vector

  def get_insult_misspellings(self, isTest=True):
    if isTest:
      fileName = self.insultTestSplitSpaces
    else:
      fileName = self.insultSplitSpaces
    vector = []
    for sent in fileName:
      misspelled_words = 0
      total_words = 0
      for word in sent:
        word = word.strip()
        #skip numbers, punctuations, non-words(i guess this is my splitting?), and usernames
        if word.isdigit() or word in self.punctuation or len(word) == 0 or word[0] == '@':
          continue
        total_words += 1
        if not self.dictionary.check(word):
          misspelled_words += 1
      if total_words == 0:
        continue
      vector.append(misspelled_words/float(total_words))
    return vector

if __name__ == '__main__':
  ms = Misspellings()
  clean_vector = ms.get_clean_misspellings()
  insult_vector = ms.get_insult_misspellings()
  print sum(clean_vector)/len(clean_vector)
  print sum(insult_vector)/len(insult_vector)
  print np.std(clean_vector)
  print np.std(insult_vector)



  


