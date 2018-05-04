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


################################################################################
# getting results from success output files
################################################################################

def get_results(fpath):
    found = 0
    for line in get_lines_in_file(fpath):
        line = line.strip()
        if line.startswith("Feature"):

            _, feature_name = line.split()

        elif line.startswith("Test Accuracy"):
            _, accuracy = line.split('=')
        elif line.startswith("avg / total"):
            values = line.split()
            precision, recall, fscore = values[-4], values[-3], values[-2]

            found = 1
            break
    if found:
        return feature_name, accuracy, precision, recall, fscore
    else:
        return None, None, None, None, None




def get_results_other(fpath):
    found = 0
    for line in get_lines_in_file(fpath):
        line = line.strip()
        if line.startswith("Feature"):

            _, feature_name = line.split()

        elif line.startswith("Avg Accuracy"):
            _, accuracy = line.split('=')

        elif line.startswith('Macro Precision Score'):
            vals_p=line.split(',')
            precision_macro, precision_micro,precision_weighted=vals_p[1],vals_p[3],vals_p[5]
        elif line.startswith('Macro Recall score'):
            vals_r = line.split(',')
            recall_macro, recall_micro, recall_weighted =vals_r[1], vals_r[3], vals_r[5]

        elif line.startswith('Macro F1-score'):
            vals_f = line.split(',')
            # print vals_f
            f_macro, f_micro, f_weighted =vals_f[0] .split(' ')[-2],vals_f[1].split(' ')[-2],vals_f[2].split(' ')[-1][:-1]
            # print f_macro,f_micro,f_weighted
    return feature_name,accuracy,precision_macro,recall_macro,f_macro,precision_micro,recall_micro,f_micro,precision_weighted,recall_weighted,f_weighted

import matplotlib.pyplot as plt
import numpy as np

#TODO in analysis
#REF andress muller tutorial
def visualize_coefficients(classifier, feature_names, n_top_features=25):
    # get coefficients with large absolute values
    coef = classifier.coef_.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[interesting_coefficients]]
    plt.bar(np.arange(50), coef[interesting_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 51), feature_names[interesting_coefficients], rotation=60, ha="right");


def display_success_results(result_dir):
    for feature in current_app._get_current_object().config['FEATURES_IN_USE']:
        if isinstance(feature, list):
            feature = "-".join(feature)
        feature_name, accuracy, precision, recall, fscore,precision_m, recall_m, fscore_m,precision_w, recall_w, fscore_w = get_results_other(os.path.join(result_dir, feature + '.txt'))
        print ('{feature},{accuracy},{precision},{recall},{fscore},{precision_m},{recall_m},{fscore_m},{precision_w},{recall_w},{fscore_w}'.format(feature=feature_name, accuracy=accuracy,
                                                                               precision=precision, recall=recall,
                                                                               fscore=fscore,precision_m=precision_m, recall_m=recall_m,
                                                                               fscore_m=fscore_m,precision_w=precision_w, recall_w=recall_w,
                                                                               fscore_w=fscore_w))






def collect_results(output_order,result_dir,type='clf'):
    '''
     utility to collect results from files
    :param result_dir: The directory that contains all the results files
    :return: None, prints the results in format feature name and then scores all separated by commas
    '''

    result_dic={}
    for result_file in os.listdir(result_dir):
        # print(result_file)
        with open(os.path.join(result_dir, result_file), 'r') as f_in:
            lines=f_in.readlines()
            results = lines[-1]
            for line in lines:
                if line.startswith('Feature'):
                    start_line=line
                    break

            if type=='reg':
                 _,feature_name=start_line.split('Feature')
            elif type=='clf':
                _, feature_name = start_line.split('Feature')
            else:
                raise NotImplementedError("Type not known")
            feature_name=feature_name.strip()
            output="{feature_name},{scores}".format(feature_name=feature_name, scores=results)
            if feature_name.startswith('['):
                #eg  ['concepts_score', 'concepts', 'writing_density_scaled', 'categorical_char_ngram_mid_word']
                feature_name=feature_name.lstrip('[').rstrip(']')
                key="-".join( [ feature.strip().strip("'") for feature in feature_name.split(',')])
                result_dic[key]=output
            else:
                result_dic[feature_name]=output
    # pprint(result_dic)
    print (set(result_dic.keys())-set([ "-".join(f) if isinstance(f,list) else f  for f in output_order ]))
    for f in output_order:
        # print (f)
        if isinstance(f,list):
            key="-".join(f)
            print (result_dic.get(key,''))
        else:
            print (result_dic.get(f,''))







if __name__ == '__main__':
    base_path='/home/suraj/resouces/remaining/'
    for file in os.listdir(base_path):
        if file=='099161853X_099161853X.txt':
            sentences = get_n_sentences(os.path.join(base_path,file), 1000,encoding='utf-8')
            with codecs.open(os.path.join('/home/suraj/resouces/remaining/',file+'_1000'),'w',encoding='utf-8') as f_out:
                f_out.write(sentences)
