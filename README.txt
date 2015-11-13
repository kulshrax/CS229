~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FILES / DIRECTORIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- data directory
	This contains raw data, for now just test.csv and train.csv (from Kaggle).

- corpora directory
	This contains text files used for the LanguageModel class. They consist of
	individual sentences, one per line, in plaintext. There are four,
	{clean, insult}_corpus_{train, test}.txt

- LanguageModel.py
	Contains a shell class used to easily interact with generating new NLTK
	LanguageModels. Just hides a lot of the complexity of the NLTK library, and
	will probably be extended when we stop using simple Unigrams later.

- naiveBayesBaseline.py
	Contains the baseline version of the naiveBayes algorithm.
	There are several optimizations on top of the baseline implementation
	that can be switched off with flags. See the RESULTS.txt file for details
	on this.

- RESULTS.txt 
	Contains the printouts from individual runs. In here, I noted the different
	things I tried, how they work, and how to modify naiveBayesBaseline.py to
	recreate the results.



Ideas for Improvements:
- Kneser-ney
- Spell check
- DONE Laplace Smoothing
- DONE Remove stopwords