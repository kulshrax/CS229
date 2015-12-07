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

    self.cleanTestSents = LanguageModel(CLEAN_TEST_FILE).splitBySpaces()
    self.insultTestSents = LanguageModel(INSULT_TEST_FILE).splitBySpaces()

  def get_sentiments(self, sents):
    scores = {}
    num_comment = 0
    for comment in sents:
      num_comment += 1
      full_comment = ' '.join(comment)
      scores[num_comment] = senti_classifier.polarity_scores([full_comment])
      print num_comment
      print scores[num_comment]
    return scores

if __name__ == '__main__':
  s = Sentiment()
  training_clean = s.get_sentiments(s.cleanSents)
  training_insult = s.get_sentiments(s.insultSents)
  test_clean = s.get_sentiments(s.cleanTestSents)
  test_insult = s.get_sentiments(s.insultTestSents)



