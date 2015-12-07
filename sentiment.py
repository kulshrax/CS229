from senti_classifier import senti_classifier
from LanguageModel import LanguageModel
import subprocess
import pickle
import unirest
import requests

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

  def api_sentiments(self, sents):
    scores = []
    num_comment = 0
    for comment in sents:
      num_comment += 1
      full_comment = ' '.join(comment)
      #json_return = eval(subprocess.check_output(["curl", "--get", "--include", 'https://twinword-sentiment-analysis.p.mashape.com/analyze/?text=great+value+in+its+price+range!', "-H", 'X-Mashape-Key: MoIfj3DC1XmshWnfXeSDpSpIydB0p1qH8lqjsnEsOWTHDi1lTL', "-H", 'Accept: application/json']))
      resp = requests.get('https://twinword-sentiment-analysis.p.mashape.com/analyze/?text=%s'%(full_comment), headers={"X-Mashape-Key": "MoIfj3DC1XmshWnfXeSDpSpIydB0p1qH8lqjsnEsOWTHDi1lTL", "Accept": "application/json"}, verify=False)
      try:
        s = resp.json()['score']
      except ValueError:
        s = 0.0
      print s
      print num_comment
      scores.append(s)
    return scores

  def get_clean_train_vector(self):
    vector = []
    f = open('data/clean_train_sentiment.csv', 'r')
    for line in f:
      vector.append(float(line))
    return vector

  def get_insult_train_vector(self):
    vector = []
    f = open('data/insult_train_sentiment.csv', 'r')
    for line in f:
      vector.append(float(line))
    return vector

  def get_clean_test_vector(self):
    vector = []
    f = open('data/clean_test_sentiment.csv', 'r')
    for line in f:
      vector.append(float(line))
    return vector

  def get_insult_test_vector(self):
    vector = []
    f = open('data/insult_test_sentiment.csv', 'r')
    for line in f:
      vector.append(float(line))
    return vector

if __name__ == '__main__':
  s = Sentiment()

  clean_train_scores = s.api_sentiments(s.cleanSents)
  clean_train_file = open('data/clean_train_sentiment.csv', 'w')
  for elem in clean_train_scores:
    clean_train_file.write(str(elem)+'\n')
  clean_train_file.close()

  insult_train_scores = s.api_sentiments(s.insultSents)
  insult_train_file = open('data/insult_train_sentiment.csv', 'w')
  for elem in insult_train_scores:
    insult_train_file.write(str(elem)+'\n')
  insult_train_file.close()

  clean_test_scores = s.api_sentiments(s.cleanTestSents)
  clean_test_file = open('data/clean_test_sentiment.csv', 'w')
  for elem in clean_test_scores:
    clean_test_file.write(str(elem)+'\n')
  clean_test_file.close()

  insult_test_scores = s.api_sentiments(s.insultTestSents)
  insult_test_file = open('data/insult_test_sentiment.csv', 'w')
  for elem in insult_test_scores:
    insult_test_file.write(str(elem)+'\n')
  insult_test_file.close()

  print s.get_clean_test_vector()
  print s.get_insult_test_vector()
  print s.get_clean_train_vector()
  print s.get_insult_train_vector()





