# -*- coding: utf-8 -*-

import os
import joblib
from mongoengine import Q
from features import create_feature
from features.phonetic import generate_phonetic_representation, get_stress_markers
from manage import app
from loader import extract, pos_data, load_concepts
from sklearn import preprocessing
from bookweb.models import *
from functools import wraps
from bookweb import celery
from nltk import sent_tokenize
import itertools

__author__ = 'suraj'

# A lazy decorator
def lazy(func):
    """ A decorator function designed to wrap attributes that need to be
        generated, but will not change. This is useful if the attribute is
        used a lot, but also often never used, as it gives us speed in both
        situations.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        name = "_" + func.__name__
        try:
            return getattr(self, name)
        except AttributeError:
            value = func(self, *args, **kwargs)
            setattr(self, name, value)
            return value

    return wrapper


class BookDataWrapper(object):
    """
    Wraps essential data for features extraction

    """

    _POS_PATH = app.config['STANFORD_PARSER_OUTPUT']
    _SENTIC_PATH = app.config['SENTIC_PARSE_OUTPUT']

    # _POS_PATH = app.config['STANFORD_PARSER_OUTPUT']
    # _SENTIC_PATH = app.config['SENTIC_PARSE_OUTPUT']

    def __init__(self, book_id, isbn_10, book_title, genre, similar_book_tags=[], content=''):
        self.book_id = book_id
        self.isbn_10 = isbn_10
        self.content = content
        self.book_title = book_title
        self.genre = genre
        self.similar_books_tags = similar_book_tags

    def _get_parse_file_name(self, parsed_path, ext='_st_parser.txt'):
        files = os.listdir(parsed_path)
        if self.isbn_10:
            st_parse_file_name = self.book_id + '_' + self.isbn_10 + ext
        else:
            st_parse_file_name = self.book_id + ext
        if not os.path.exists(os.path.join(parsed_path, st_parse_file_name)):
            for b in files:
                gid_isbn = b.replace(ext, '')
                gid = gid_isbn.rsplit('_', 1)[0]
                if self.book_id == gid:
                    st_parse_file_name = b
                    break
        return st_parse_file_name

    def read_st_parse_file(self):
        st_parse_file_name = self._get_parse_file_name(self._POS_PATH)
        self._word_pos, self._parse_tree, dependency = extract(os.path.join(self._POS_PATH, st_parse_file_name))
        self._pos_tag = pos_data(self._word_pos)
        if hasattr(self, 'size'):
            self._pos_tag = '\n'.join(self._pos_tag.split('\n')[:self.size])
            self._parse_tree = self._parse_tree[:self.size]

    def read_sentic_concepts(self):
        sentic_parse_file_name = self._get_parse_file_name(self._SENTIC_PATH, ext='_st_parser.txt.json')
        self._concepts, self._concepts_ls, self._sensitivity, self._attention, self._pleasantness, self._aptitude, self._polarity = load_concepts(
            os.path.join(self._SENTIC_PATH, sentic_parse_file_name))
        if hasattr(self, 'size'):
            self._sensitivity = self._sensitivity[:self.size]
            self._attention = self._attention[:self.size]
            self._pleasantness = self._pleasantness[:self.size]
            self._aptitude = self._aptitude[:self.size]
            self._polarity = self._polarity[:self.size]
            self._concepts = " ".join(list(itertools.chain.from_iterable(self._concepts_ls[:self.size])))

    def phonetic_representation(self):
        self._phonetics, self._phonetic_sent_words = generate_phonetic_representation(self.content)

    def stress_represenation(self):
        self._stress_markers = get_stress_markers(self.content, two_classes_only=True)

    def all_stress_represenation(self):
        self._all_stress_markers = get_stress_markers(self.content, two_classes_only=False)

    def of_size(self, size):
        content = ' '.join(sent_tokenize(self.content)[:size])
        sub_book = BookDataWrapper(book_id=self.book_id, isbn_10=self.isbn_10, book_title=self.book_title,
                                   genre=self.genre, similar_book_tags=[], content=content)
        setattr(sub_book, 'size', size)  # set the size attribute for the sub object

        return sub_book

    @property
    @lazy
    def concepts(self):
        self.read_sentic_concepts()
        return self._concepts

    @property
    @lazy
    def sensitivity(self):
        self.read_sentic_concepts()
        return self._sensitivity

    @property
    @lazy
    def attention(self):
        self.read_sentic_concepts()
        return self._attention

    @property
    @lazy
    def aptitude(self):
        self.read_sentic_concepts()
        return self._aptitude

    @property
    @lazy
    def polarity(self):
        self.read_sentic_concepts()
        return self._polarity

    @property
    @lazy
    def pleasantness(self):
        self.read_sentic_concepts()
        return self._pleasantness

    @property
    @lazy
    def pos_tag(self):
        self.read_st_parse_file()
        return self._pos_tag

    @property
    @lazy
    def parse_tree(self):
        self.read_st_parse_file()
        return self._parse_tree

    @property
    @lazy
    def phonetics(self):
        self.phonetic_representation()
        return self._phonetics

    @property
    @lazy
    def phonetic_sent_words(self):
        self.phonetic_representation()
        return self._phonetic_sent_words

    @property
    @lazy
    def word_pos(self):
        self.read_st_parse_file()
        return self._word_pos

    @property
    @lazy
    def stress_markers(self):
        self.stress_represenation()
        return self._stress_markers

    @property
    @lazy
    def all_stress_markers(self):
        self.all_stress_represenation()
        return self._all_stress_markers


def dump(obj, file):
    with open(file, 'w') as f:
        joblib.dump(obj, file)


def create_vectors(feature_name, data, dump_dir):
    X, X_id = [], []
    feature_name, feature_obj = create_feature(feature_name)
    for book in data:
        X.append(book)
        X_id.append(book.book_id)
    print(len(X))

    V = feature_obj.fit_transform(X)
    if feature_name in ['writing_density', 'readability']:
        V = preprocessing.scale(V)
    # dump the features vector

    dump(V, os.path.join(dump_dir, feature_name + '.vector'))
    dump(feature_obj, os.path.join(dump_dir, feature_name + '.model'))
    dump(X_id, os.path.join(dump_dir, feature_name + '.books'))


def prepare_vectors(books=None):
    data = []

    # for book in Book.objects(status=BookStatus.UNPROCESSED):
    for book in Book.objects(Q(status=BookStatus.UNPROCESSED) | Q(is_active=True)):
        content = book.content.replace('\n', ' ').replace('\r', '').replace('\x0C', '')
        if books:
            if book.book_id in books:
                data.append(
                    BookDataWrapper(book_id=book.book_id, book_title=book.title, isbn_10=book.isbn_10, content=content,
                                   genre=book.genre, similar_book_tags=[]))
        else:
            data.append(
                BookDataWrapper(book_id=book.book_id, book_title=book.title, isbn_10=book.isbn_10, content=content,
                                   genre=book.genre, similar_book_tags=[]))


    print("Total data points {}".format(len(data)))
    print app.config['FEATURES']
    for feature_name in app.config['FEATURES']:
        print "For Feature ", feature_name
        create_vectors(feature_name, data, app.config['VECTORS'])

    print ('Done')


def generate_vectors(books_path, features):
    import codecs
    import pandas as pd
    def create_book_obj(path):
        book = os.path.basename(path).replace('.txt', '')
        with codecs.open(path, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
        return BookDataWrapper(book_id=book, book_title=book, isbn_10=None, content=content, genre='Fiction')

    books = [create_book_obj(os.path.join(books_path, bk)) for bk in os.listdir(books_path)]

    for feature in features:
        feature_name, feature_obj = create_feature(feature)
        V = feature_obj.fit_transform(books)
        if feature_name in ['writing_density', 'readability']:
            V = preprocessing.scale(V)
        # print V.shape
        df = pd.DataFrame(V, columns=feature_obj.get_feature_names().tolist(), index=[book.book_id for book in books])
        df.to_csv(feature_name + '_booxby_test.tsv', sep='\t')


@celery.task(bind=True)
def build_model(self, feature_name):
    import time
    self.update_state(state='PROGRESS',
                      meta={'status': "Starting Feature Extraction for %s" % "-".join(feature_name) if isinstance(
                          feature_name, list) else feature_name})
    data = []

    time.sleep(6)
    for book in Book.objects(Q(status=BookStatus.UNPROCESSED) | Q(is_active=True)):
        data.append(BookDataWrapper(book_id=book.book_id, isbn_10=book.isbn_10, content=book.content, book_title=book.title,
                                genre=book.genre, similar_book_tags=[]))
    create_vectors(feature_name, data, app.config['STAGGED_VECTORS'])

    self.update_state(state='SUCCESS',
                      meta={'status': "Done Feature Extraction for %s" % "-".join(feature_name) if isinstance(
                          feature_name, list) else feature_name})
    print ('Done')
    return True


def test():
    b = BookDataWrapper(book_id='kspZTMMVvN8C', isbn_10='030776382X', content='')
    print (b.concepts)

    print (b.sensitivity)

    print (b.parse_tree)
