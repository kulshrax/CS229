import csv
import sys

INSULT_CORPUS_FILENAME = 'corpora/insult_corpus_test.txt'
CLEAN_CORPUS_FILENAME = 'corpora/clean_corpus_test.txt'

def main(filename):
	insult_file = open(INSULT_CORPUS_FILENAME, 'w+')
	clean_file = open(CLEAN_CORPUS_FILENAME, 'w+')

	with open(filename, 'r') as csvfile:
		rdr = csv.reader(csvfile)
		for row in rdr:
			# clean row
			if row[0] == '0':
				pass
				clean_file.write(row[2] + "\n")			
			# insult row
			else:
				pass
				insult_file.write(row[2] + "\n")

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print ("Usage: python create_corpora.py filename.csv")
		exit(0)
	main(sys.argv[1])
