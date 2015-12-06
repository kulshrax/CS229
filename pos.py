#!/usr/bin/env python

"""
Simple wrapper around NLTK's part of speech tagger for finding
the fraction of nouns, verbs, and adjectives in a given corpus.
"""

from __future__ import division
from collections import Counter
import nltk

TAG_CLASSES = {
    'N': ['NN', 'NNS', 'NNP', 'NNPS'], # Nouns
    'V': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], # Verbs
    'A': ['JJ', 'JJR', 'JJS'], # Adjectives
}

def simplify_tags(tagged_tokens, tag_classes=TAG_CLASSES):
    """Convert NTLK's part of speech tags to simpler, more general tags.
       Omit any token that doesn't belong to any of the given tag classes.
    """
    tag_to_simple = {}
    for simple_tag, tags in tag_classes.iteritems():
        for tag in tags:
            tag_to_simple[tag] = simple_tag

    return [(token, tag_to_simple[tag]) 
            for token, tag in tagged_tokens
            if tag in tag_to_simple]

def pos_counts(tokens):
    """Count the number of nouns, verbs, and adjectives in the token list."""
    tagged = nltk.pos_tag(tokens)
    simplified = simplify_tags(tagged)
    return Counter(tag for _, tag in simplified)

def pos_fractions(tokens):
    """Return the fraction of nouns, verbs, and adjectives in the token list."""
    counts = pos_counts(tokens)
    total = len(tokens)
    return dict((tag, count/total) for tag, count in counts.iteritems())