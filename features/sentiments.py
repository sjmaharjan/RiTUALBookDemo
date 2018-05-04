# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict
import csv
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'suraj'

__all__ = ['SentiWordNetFeature', 'SenticConceptsTfidfVectorizer', 'SenticConceptsScores']


# REF http://sentiwordnet.isti.cnr.it/code/SentiWordNetDemoCode.java
# REF Building Machine Learning Systems with Python Section Sentiment analysis
def load_sentiwordnet(path):
    scores = defaultdict(list)
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            # skip comments
            if line[0].startswith("#"):
                continue
            if len(line) == 1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            # print POS,PosScore,NegScore,SynsetTerms
            for term in SynsetTerms.split(" "):
                # drop number at the end of every term
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term.split("#")[0])
                scores[key].append((float(PosScore), float(NegScore)))
    for key, value in scores.items():
        scores[key] = np.mean(value, axis=0)
    return scores


# REF Building Machine Learning Systems with Python Section Sentiment analysis
class SentiWordNetFeature(BaseEstimator):
    def __init__(self):
        self.sentiwordnet = load_sentiwordnet('resources/SentiWordNet_3.0.0_20130122.txt')

    def get_feature_names(self):
        return np.array(['sent_neut', 'sent_pos', 'sent_neg', 'nouns', 'adjectives', 'verbs', 'adverbs'])

    def _get_sentiments(self, d):
        tagged_sent = d.tagged_data
        pos_vals = []
        neg_vals = []
        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.
        for tag in tagged_sent.split():
            sent_len = 0
            t, p, c = tag.rsplit('/', 2)
            p_val, n_val = 0, 0
            sent_pos_type = None
            if p.startswith("NN"):
                sent_pos_type = "n"
                nouns += 1
            elif p.startswith("JJ"):
                sent_pos_type = "a"
                adjectives += 1
            elif p.startswith("VB"):
                sent_pos_type = "v"
                verbs += 1
            elif p.startswith("RB"):
                sent_pos_type = "r"
                adverbs += 1
            if sent_pos_type is not None:
                sent_word = "%s/%s" % (sent_pos_type, t.lower())
                if sent_word in self.sentiwordnet:
                    p_val, n_val = self.sentiwordnet[sent_word]
            pos_vals.append(p_val)
            neg_vals.append(n_val)
            sent_len += 1

        l = sent_len
        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)
        return [1 - avg_pos_val - avg_neg_val, avg_pos_val, avg_neg_val, nouns / l, adjectives / l, verbs / l,
                adverbs / l]

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        X = np.array([self._get_sentiments(d) for d in documents])
        return X


# Sentic Concepts Features




class SenticConceptsTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.concepts))


class SenticConceptsScores(BaseEstimator, TransformerMixin):
    def get_feature_names(self):
        return np.array(
            ['avg_sensitivity', 'avg_attention', 'avg_pleasantness', 'avg_aptitude', 'avg_polarity',
             #        'max_sensitivity',
             # 'max_attention', 'max_pleasantness', 'max_aptitude', 'max_polarity', 'min_sensitivity', 'min_attention',
             # 'min_pleasantness', 'min_aptitude', 'min_polarity'
             ])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        feature_vector = []
        for doc in documents:
            avg_sensitivity = np.mean(doc.sensitivity) if doc.sensitivity else 0.
            # min_sensitivity = np.min(doc.sensitivity)
            # max_sensitivity = np.max(doc.sensitivity)

            avg_attention = np.mean(doc.attention) if doc.attention  else 0.
            # min_attention = np.min(doc.attention)
            # max_attention = np.max(doc.attention)

            avg_pleasantness = np.mean(doc.pleasantness) if doc.pleasantness else 0.
            # min_pleasantness = np.min(doc.pleasantness)
            # max_pleasantness = np.max(doc.pleasantness)

            avg_aptitude = np.mean(doc.aptitude) if doc.aptitude else 0.
            # min_aptitude = np.min(doc.aptitude)
            # max_aptitude = np.max(doc.aptitude)

            avg_polarity = np.mean(doc.polarity) if doc.polarity  else 0.
            # min_polarity = np.min(doc.polarity)
            # max_polarity = np.max(doc.polarity)

            feature_vector.append(
                [avg_sensitivity, avg_attention, avg_pleasantness, avg_aptitude, avg_polarity,
                 # max_sensitivity,
                 # max_attention, max_pleasantness, max_aptitude, max_aptitude, max_polarity, min_attention,
                 # min_pleasantness, min_aptitude, min_aptitude, min_polarity
                 ])

        return np.array(feature_vector)
