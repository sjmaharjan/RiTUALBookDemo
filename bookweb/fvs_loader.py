__author__ = 'suraj'

import joblib
import os
from flask import current_app
from gensim.models import Word2Vec
import numpy as np


class FeatureLoader(object):
    """Loads the dumped features objects, vectors and ids

        ids and vectors are parallel array
        ids contains the unique names/id of the book
        vectors contains the corresponding features vector for the book

        Follow lazy loading scheme



    """
    def __init__(self, feature, config):
        self.load_dir = config['VECTORS']
        self._feature = feature

    def model(self):
        if not hasattr(self, 'model_'):
            self.model_ = joblib.load(os.path.join(self.load_dir, self._feature + '.model'))
        return self.model_

    def vectors(self):
        if not hasattr(self, 'vectors_'):
            print "Loading vectors ....."
            self.vectors_ =  joblib.load(os.path.join(self.load_dir, self._feature + '.vector'))
        return self.vectors_


    def ids(self):
        if not hasattr(self, 'ids_'):
            self.ids_ = joblib.load(os.path.join(self.load_dir, self._feature + '.books'))
        return self.ids_


#
# class ProxyFeatureLoader(object):
#     def __init__(self,):
#         self.__implementation = FeatureLoader()
#
#     def __getattr__(self, name):
#         return getattr(self.__implementation, name)


class FeatureInitializer(object):
    """
    Initialize features vectors with the app

    """


    def __init__(self, app=None, config=None):
        if app is not None:
            self.init_app(app, config)

    def init_app(self, app, config=None):

        app.feature_extensions = getattr(app, 'feature_extensions', {})

        if not 'featurevectors' in app.feature_extensions:
            app.feature_extensions['featurevectors'] = {}

        if self in app.feature_extensions['featurevectors']:
            raise Exception('Feature Extension already initialized')

        if not config:
            # If not passed a config then we read the features settings
            # from the app config.
            config = app.config

        features_to_load = config['FEATURES']
        load_dir = config['VECTORS']
        myFeatures={}
        features_dictionary = {}
        for feature in features_to_load:
            if isinstance(feature, list):
                feature = "-".join(feature)
            myFeatures[feature]=joblib.load(os.path.join(load_dir, feature + '.vector'))
            if os.path.exists(os.path.join(load_dir, feature + '.vector')):
                print "Loding features", feature
                # features_dictionary[features] = {'model': joblib.load(os.path.join(load_dir, features + '.model')),
                #                                 'vectors': joblib.load(os.path.join(load_dir, features + '.vector')),
                #                                 'ids': joblib.load(os.path.join(load_dir, features + '.books'))}
                features_dictionary[feature] =FeatureLoader(feature,config)
        print 'Done loading features vectors'

        app.feature_extensions['featurevectors'][self] = {'app': app,'features': features_dictionary}
        app.myFeature=myFeatures

    @property
    def feature_vectors(self):
        return current_app.feature_extensions['featurevectors'][self]['features']






class EmbeddingLoader(object):
    """Loads the experiential language embeddings
    """
    def __init__(self, config):
        self.exp_lang_path = config['EXPLANGEMB']

    def load_embedding(self):
        self._exp_lang_emb = Word2Vec.load(self.exp_lang_path)
        self._dimension = self._exp_lang_emb.syn0.shape[1]
        self._index2word_set = set(self._exp_lang_emb.index2word)

    def get_emb_size(self):
        if not hasattr(self, '_exp_lang_emb'):
            self.load_embedding()
        return self._dimension



    def phrase_embedding(self,words):
        if not hasattr(self, '_exp_lang_emb'):
            self.load_embedding()

        emb = np.zeros((self._dimension,), dtype=np.float32)

        nwords = 0.

        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its embedding vector to the total
        for word in words.split():
            word=word.lower()
            if word in self._index2word_set:
                nwords = nwords + 1.
                emb = np.add(emb, self._exp_lang_emb[word])
        #
        # Divide the result by the number of words to get the average
        if nwords>0.0:
            emb = np.divide(emb, nwords)

        return emb

