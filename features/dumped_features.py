# -*- coding: utf-8 -*-
from __future__ import division, print_function
from flask import current_app
from manage import app
from sklearn.base import BaseEstimator, TransformerMixin
import os
from bookweb import fvs
import numpy as np
import scipy.sparse as sp

__author__ = 'suraj'

__all__ = ['DumpedFeaturesTransformers']

class DumpedFeaturesTransformers(BaseEstimator, TransformerMixin):
    """
    Loads the dumped features

    """
    # __dumped_dir = current_app.config['VECTORS']
    __dumped_dir = app.config['VECTORS']

    def __init__(self, feature):
        self.feature = feature

        if os.path.exists(os.path.join(self.__dumped_dir, self.feature + '.vector')):

            # self._X_ids = joblib.load(os.path.join(self.__dumped_dir, self.features + '.books'))
            # self._vectors = joblib.load(os.path.join(self.__dumped_dir, self.features + '.vector'))
            # self._model = joblib.load(os.path.join(self.__dumped_dir, self.features + '.model'))



            self._X_ids = fvs.feature_vectors[self.feature].ids()
            self._vectors = fvs.feature_vectors[self.feature].vectors()
            self._model = fvs.feature_vectors[self.feature].model()

        else:
            raise ValueError("Feature dump for  %s does not exist in %s" % (
                feature, os.path.join(self.__dumped_dir, feature + '.vector')))

    def get_feature_names(self):
        return self._model.get_feature_names()

    def fit(self, X, y=None):
        return self

    def transform(self, books):
        X = []
        sparse = sp.issparse(self._vectors)
        for book in books:

            if book.book_id in self._X_ids:
                book_index = self._X_ids.index(book.book_id)
                if sparse:
                    X.append(self._vectors[book_index].toarray()[0])
                else:
                    X.append(self._vectors[book_index])
            else:
                # this should not happen
                print ("Herer inside danger zone")
                X.append(self._model.transform(book)[0])

        if sparse:
            # print X[0]

            X = sp.csr_matrix(X)
        else:
            X = np.array(X)
        # print X[0]
        return X
