from LanguageModel import LanguageModel


INSULT_TRAIN_FILE = 'insult_corpus_train.txt'
CLEAN_TRAIN_FILE = 'clean_corpus_train.txt'

INSULT_TEST_FILE = 'insult_corpus_test.txt'
CLEAN_TEST_FILE = 'clean_corpus_test.txt'

def main():
	cleanLM = LanguageModel(CLEAN_TRAIN_FILE)
	insultLM = LanguageModel(INSULT_TRAIN_FILE)


if __name__ == "__main__":
	main()
