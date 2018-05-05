# -*- coding: utf-8 -*-
import codecs

import nltk

from bookweb.models import Book
import pandas as pd
from contextlib import contextmanager
import sys
import re
import os
from flask import current_app
from pprint import pprint

__author__ = 'suraj'


# Ref :Raymond Hettinger beautiful python code slides
# Python 3 has this features but python 2 does not
@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


def get_files_in_folder(folder):
    all_files = sorted(os.listdir(folder))
    if ".DS_Store" in all_files:
        all_files.remove(".DS_Store")
    return all_files


def get_lines_in_file(fpath):
    with open(fpath) as fhandle:
        return fhandle.readlines()




def get_n_sentences(filename, n=1000,encoding='latin-1'):
    full_text = ""
    with codecs.open(filename, 'r',encoding=encoding) as f_in:
        for line in f_in.readlines():
            line = line.strip()
            if not line:
                continue
            sentences = nltk.sent_tokenize(line)
            full_text += " ".join(sentences[:n-len(nltk.sent_tokenize(full_text))]) + "\n"
            if len(nltk.sent_tokenize(full_text)) >= n:
                break
    return full_text




def get_n_sentences_gutenberg(filename, n=1000,encoding='latin-1'):
    full_text = ""
    with codecs.open(filename, 'r',encoding=encoding) as f_in:
        for line in f_in.readlines():
            line = line.strip()
            if not line:
                full_text += "\n\n"
                continue
            sentences = nltk.sent_tokenize(line)
            full_text += " ".join(sentences[:n-len(nltk.sent_tokenize(full_text))]) + " "
            if len(nltk.sent_tokenize(full_text)) >= n:
                break
    return full_text


def get_all_books(filename):
    book_data = []
    for book in Book.objects(is_active=True):
        book_data.append({'book_id': book.book_id, 'book_title': book.title.title(), 'book_isbn_10': book.isbn_10,
                          'book_isbn_13': book.isbn_13})

    df = pd.DataFrame(book_data)
    df.to_csv(filename, sep='\t', encoding='utf-8',index=False)


