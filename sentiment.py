from senti_classifier import senti_classifier
from LanguageModel import LanguageModel

class Sentiment(object):

  def __init__(self):

    INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
    CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

    INSULT_TEST_FILE = 'insult_corpus_test.txt'
    CLEAN_TEST_FILE = 'clean_corpus_test.txt'

    self.cleanSents = LanguageModel(CLEAN_TRAIN_FILE).splitBySpaces()
    self.insultSents = LanguageModel(INSULT_TRAIN_FILE).splitBySpaces()

  def get_train_clean_sentiments(self):
    for word in self.cleanSents:
      pos_score, neg_score = senti_classifier.polarity_scores(word)
      print pos_score
      print neg_score

  def get_train_insult_sentiments(self):
    return senti_classifier.polarity_scores(self.insultSents)

if __name__ == '__main__':
  s = Sentiment()
  print 'Clean:'
  print s.get_train_clean_sentiments()
  print 'Insult:'
  print s.get_train_insult_sentiments()

