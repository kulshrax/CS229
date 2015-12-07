#!/usr/bin/env python
import json
import math
import os
import re
import sys

import math
import collections
from PorterStemmer import PorterStemmer


class IRSystem:

    def __init__(self):
        # For holding the data - initialized in read_data()
        self.titles = []
        self.docs = []
        self.vocab = []
        # For the text pre-processing.
        self.alphanum = re.compile('[^a-zA-Z0-9]')
        self.p = PorterStemmer()


    def get_uniq_words(self):
        uniq = set()
        for doc in self.docs:
            for word in doc:
                uniq.add(word)
        return uniq


    def __read_raw_data(self, dirname):
        print "Stemming Documents..."

        titles = []
        docs = []
        os.mkdir('%s/stemmed' % dirname)
        title_pattern = re.compile('(.*) \d+\.txt')

        # make sure we're only getting the files we actually want
        filenames = []
        for filename in os.listdir('%s/raw' % dirname):
            if filename.endswith(".txt") and not filename.startswith("."):
                filenames.append(filename)

        for i, filename in enumerate(filenames):
            title = title_pattern.search(filename).group(1)
            print "    Doc %d of %d: %s" % (i+1, len(filenames), title)
            titles.append(title)
            contents = []
            f = open('%s/raw/%s' % (dirname, filename), 'r')
            of = open('%s/stemmed/%s.txt' % (dirname, title), 'w')
            for line in f:
                # make sure everything is lower case
                line = line.lower()
                # split on whitespace
                line = [xx.strip() for xx in line.split()]
                # remove non alphanumeric characters
                line = [self.alphanum.sub('', xx) for xx in line]
                # remove any words that are now empty
                line = [xx for xx in line if xx != '']
                # stem words
                line = [self.p.stem(xx) for xx in line]
                # add to the document's conents
                contents.extend(line)
                if len(line) > 0:
                    of.write(" ".join(line))
                    of.write('\n')
            f.close()
            of.close()
            docs.append(contents)
        return titles, docs


    def __read_stemmed_data(self, dirname):
        print "Already stemmed!"
        titles = []
        docs = []

        # make sure we're only getting the files we actually want
        filenames = []
        for filename in os.listdir('%s/stemmed' % dirname):
            if filename.endswith(".txt") and not filename.startswith("."):
                filenames.append(filename)

        if len(filenames) != 60:
            msg = "There are not 60 documents in ../data/RiderHaggard/stemmed/\n"
            msg += "Remove ../data/RiderHaggard/stemmed/ directory and re-run."
            raise Exception(msg)

        for i, filename in enumerate(filenames):
            title = filename.split('.')[0]
            titles.append(title)
            contents = []
            f = open('%s/stemmed/%s' % (dirname, filename), 'r')
            for line in f:
                # split on whitespace
                line = [xx.strip() for xx in line.split()]
                # add to the document's conents
                contents.extend(line)
            f.close()
            docs.append(contents)

        return titles, docs


    def read_data(self, dirname):
        """
        Given the location of the 'data' directory, reads in the documents to
        be indexed.
        """
        # NOTE: We cache stemmed documents for speed
        #       (i.e. write to files in new 'stemmed/' dir).

        print "Reading in documents..."
        # dict mapping file names to list of "words" (tokens)
        filenames = os.listdir(dirname)
        subdirs = os.listdir(dirname)
        if 'stemmed' in subdirs:
            titles, docs = self.__read_stemmed_data(dirname)
        else:
            titles, docs = self.__read_raw_data(dirname)

        # Sort document alphabetically by title to ensure we have the proper
        # document indices when referring to them.
        ordering = [idx for idx, title in sorted(enumerate(titles),
            key = lambda xx : xx[1])]

        self.titles = []
        self.docs = []
        numdocs = len(docs)
        for d in range(numdocs):
            self.titles.append(titles[ordering[d]])
            self.docs.append(docs[ordering[d]])

        # Get the vocabulary.
        self.vocab = [xx for xx in self.get_uniq_words()]


    def compute_tfidf(self):
        # -------------------------------------------------------------------
        # TODO: Compute and store TF-IDF values for words and documents.
        #       Recall that you can make use of:
        #         * self.vocab: a list of all distinct (stemmed) words
        #         * self.docs: a list of lists, where the i-th document is
        #                   self.docs[i] => ['word1', 'word2', ..., 'wordN']
        #       NOTE that you probably do *not* want to store a value for every
        #       word-document pair, but rather just for those pairs where a
        #       word actually occurs in the document.
        print "Calculating tf-idf..."
        self.tfs = dict()
        self.dfs = collections.defaultdict(lambda: 0)
        self.tfidf = dict()

        for i, doc in enumerate(self.docs):
          already_counted = set()
          if i not in self.tfs.keys():
            self.tfs[i] = dict()
          
          for word in doc:
            if word not in self.tfs[i]:
              self.tfs[i][word] = 1
            else:
              self.tfs[i][word] = self.tfs[i][word] + 1

            if word not in already_counted:
              self.dfs[word] = self.dfs[word] + 1
              already_counted.add(word)

        N = len(self.docs)
        for i, doc in enumerate(self.docs):
          if i not in self.tfidf:
            self.tfidf[i] = dict()
          for word in set(doc):
            log_tf = (1.0 + math.log(self.tfs[i][word], 10))
            log_idf = math.log((N + 0.0)/self.dfs[word], 10)
            self.tfidf[i][word] = log_tf * log_idf

        # ------------------------------------------------------------------


    def get_tfidf(self, word, document):
        # ------------------------------------------------------------------
        # TODO: Return the tf-idf weigthing for the given word (string) and
        #       document index.
        tfidf = self.tfidf[document][word]
        # ------------------------------------------------------------------
        return tfidf


    def get_tfidf_unstemmed(self, word, document):
        """
        This function gets the TF-IDF of an *unstemmed* word in a document.
        Stems the word and then calls get_tfidf. You should *not* need to
        change this interface, but it is necessary for submission.
        """
        word = self.p.stem(word)
        return self.get_tfidf(word, document)


    def index(self):
        """
        Build an index of the documents.
        """
        print "Indexing..."
        # ------------------------------------------------------------------
        # TODO: Create an inverted, positional index.
        #       Granted this may not be a linked list as in a proper
        #       implementation.
        #       This index should allow easy access to both 
        #       1) the documents in which a particular word is contained, and 
        #       2) for every document, the positions of that word in the document 
        #       Some helpful instance variables:
        #         * self.docs = List of documents
        #         * self.titles = List of titles

        inv_index = dict()
        for i, doc in enumerate(self.docs):
        	for j, word in enumerate(doc):
        		if word in inv_index:
        			postings = inv_index[word]
      			else:
      				postings = dict()
      				inv_index[word] = postings
      	
      			if i in postings:
      				postings[i].append(j)
      			else:
      				indices = [j]
      				postings[i] = indices
        self.inv_index = inv_index
        # ------------------------------------------------------------------


    def get_posting(self, word):
        """
        Given a word, this returns the list of document indices (sorted) in
        which the word occurs.
        """
        # ------------------------------------------------------------------
        return self.inv_index[word].keys()
        # ------------------------------------------------------------------


    def get_posting_unstemmed(self, word):
        """
        Given a word, this *stems* the word and then calls get_posting on the
        stemmed word to get its postings list. You should *not* need to change
        this function. It is needed for submission.
        """
        word = self.p.stem(word)
        return self.get_posting(word)


    def boolean_retrieve(self, query):
        """
        Given a query in the form of a list of *stemmed* words, this returns
        the list of documents in which *all* of those words occur (ie an AND
        query).
        Return an empty list if the query does not return any documents.
        """
        # ------------------------------------------------------------------
        docs = []
        postings_sets = [set(self.get_posting(word)) for word in query]
        if len(postings_sets) == 0:
        	return []
        docs = postings_sets[0]
        for i in xrange(len(postings_sets)):
        	docs &= postings_sets[i]

        docs = list(docs)
        # ------------------------------------------------------------------

        return sorted(docs)   # sorted doesn't actually matter


    def phrase_retrieve(self, query):
        """
        Given a query in the form of an ordered list of *stemmed* words, this 
        returns the list of documents in which *all* of those words occur, and 
        in the specified order. 
        Return an empty list if the query does not return any documents. 
        """
        # ------------------------------------------------------------------
        # TODO: Implement Phrase Query retrieval (ie. return the documents 
        #       that don't just contain the words, but contain them in the 
        #       correct order) You will want to use the inverted index 
        #       that you created in index(), and may also consider using
        #       boolean_retrieve
        # Right now this just returns all possible documents!
        def phrase_retrieve_helper(inv_index, doc, query, desired_index):
        	if query == []:
        		return True	
        	if doc not in inv_index[query[0]]:
        		return False
        		
        	index_list = inv_index[query[0]][doc]
        	for index in index_list:
        		if index == desired_index:
	        		return phrase_retrieve_helper(inv_index, doc, query[1:], index+1)
	        	if index > desired_index:
	        		return False
        	return False

        docs = []
        if query == []:
        	return []
        for doc in self.get_posting(query[0]):
        	index_list = self.inv_index[query[0]][doc]
        	for index in index_list:
        		if phrase_retrieve_helper(self.inv_index, doc, query[1:], index+1):
        			docs.append(doc)
        # ------------------------------------------------------------------

        return sorted(docs)   # sorted doesn't actually matter

    def rank_retrieve(self, query):
        """
        Given a query (a list of words), return a rank-ordered list of
        documents (by ID) and score for the query.
        """
        scores = [0.0 for xx in range(len(self.docs))]
        # ------------------------------------------------------------------
        # TODO: Implement cosine similarity between a document and a list of
        #       query words.

        # Right now, this code simply gets the score by taking the Jaccard
        # similarity between the query and every document.
        
        tfs = dict()
        for word in query:
          if word not in tfs:
            tfs[word] = 1
          else:
            tfs[word] = tfs[word] + 1

        # calculate log_tfs
        for word in tfs.keys():
          tfs[word] = 1.0 + math.log(float(tfs[word]), 10)

        for d, doc in enumerate(self.docs):
          score = 0.0
          doc_weights = self.tfidf[d]
          total = math.sqrt(sum([val*val for val in doc_weights.itervalues()]))
          for word in (query):
            if word in doc_weights:
              score += float(doc_weights[word]) * tfs[word]            
          scores[d] = score / total
        # ------------------------------------------------------------------

        ranking = [idx for idx, sim in sorted(enumerate(scores),
            key = lambda xx : xx[1], reverse = True)]
        results = []
        for i in range(10):
            results.append((ranking[i], scores[ranking[i]]))
        return results


    def process_query(self, query_str):
        """
        Given a query string, process it and return the list of lowercase,
        alphanumeric, stemmed words in the string.
        """
        # make sure everything is lower case
        query = query_str.lower()
        # split on whitespace
        query = query.split()
        # remove non alphanumeric characters
        query = [self.alphanum.sub('', xx) for xx in query]
        # stem words
        query = [self.p.stem(xx) for xx in query]
        return query


    def query_retrieve(self, query_str):
        """
        Given a string, process and then return the list of matching documents
        found by boolean_retrieve().
        """
        query = self.process_query(query_str)
        return self.boolean_retrieve(query)

    def phrase_query_retrieve(self, query_str):
        """
        Given a string, process and then return the list of matching documents
        found by phrase_retrieve().
        """
        query = self.process_query(query_str)
        return self.phrase_retrieve(query)


    def query_rank(self, query_str):
        """
        Given a string, process and then return the list of the top matching
        documents, rank-ordered.
        """
        query = self.process_query(query_str)
        return self.rank_retrieve(query)


def run_tests(irsys):
    print "===== Running tests ====="

    ff = open('../data/queries.txt')
    questions = [xx.strip() for xx in ff.readlines()]
    ff.close()
    ff = open('../data/solutions.txt')
    solutions = [xx.strip() for xx in ff.readlines()]
    ff.close()

    epsilon = 1e-4
    for part in range(5):
        points = 0
        num_correct = 0
        num_total = 0

        prob = questions[part]
        soln = json.loads(solutions[part])

        if part == 0:     # inverted index test
            print "Inverted Index Test"
            words = prob.split(", ")
            for i, word in enumerate(words):
                num_total += 1
                posting = irsys.get_posting_unstemmed(word)
                if set(posting) == set(soln[i]):
                    num_correct += 1

        elif part == 1:   # boolean retrieval test
            print "Boolean Retrieval Test"
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                guess = irsys.query_retrieve(query)
                if set(guess) == set(soln[i]):
                    num_correct += 1

        elif part == 2: # phrase query test
            print "Phrase Query Retrieval"
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                guess = irsys.phrase_query_retrieve(query)
                if set(guess) == set(soln[i]):
                    num_correct += 1

        elif part == 3:   # tfidf test
            print "TF-IDF Test"
            queries = prob.split("; ")
            queries = [xx.split(", ") for xx in queries]
            queries = [(xx[0], int(xx[1])) for xx in queries]
            for i, (word, doc) in enumerate(queries):
                num_total += 1
                guess = irsys.get_tfidf_unstemmed(word, doc)
                if guess >= float(soln[i]) - epsilon and \
                        guess <= float(soln[i]) + epsilon:
                    num_correct += 1

        elif part == 4:   # cosine similarity test
            print "Cosine Similarity Test"
            queries = prob.split(", ")
            for i, query in enumerate(queries):
                num_total += 1
                ranked = irsys.query_rank(query)
                top_rank = ranked[0]
                if top_rank[0] == soln[i][0]:
                    if top_rank[1] >= float(soln[i][1]) - epsilon and \
                            top_rank[1] <= float(soln[i][1]) + epsilon:
                        num_correct += 1

        feedback = "%d/%d Correct. Accuracy: %f" % \
                (num_correct, num_total, float(num_correct)/num_total)
        if num_correct == num_total:
            points = 3
        elif num_correct > 0.75 * num_total:
            points = 2
        elif num_correct > 0:
            points = 1
        else:
            points = 0

        print "    Score: %d Feedback: %s" % (points, feedback)


def main(args):
    irsys = IRSystem()
    irsys.read_data('../data/RiderHaggard')
    irsys.index()
    irsys.compute_tfidf()

    if len(args) == 0:
        run_tests(irsys)
    else:
        query = " ".join(args)
        print "Best matching documents to '%s':" % query
        results = irsys.query_rank(query)
        for docId, score in results:
            print "%s: %e" % (irsys.titles[docId], score)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
