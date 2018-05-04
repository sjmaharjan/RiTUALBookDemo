# -*- coding: utf-8 -*-

# from gevent import monkey
# monkey.patch_all()

import os
import sys
import json
import codecs
import yaml

# Set the path
__author__ = 'suraj'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from flask_script import Manager, Command
from bookweb import create_app
from flask_script import Manager, Server, Command
from bookweb import create_app

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
manager = Manager(app)


class ExtractFeatureVectors(Command):
    def run(self):
        from dump_vectors import prepare_vectors
        # label_file = os.path.join(os.path.dirname(__file__),'resources','success_ratings.txt')
        # df = pd.read_csv(label_file, sep='\t')
        # books=set(df['book_id'].values)

        books = None
        prepare_vectors(books=books)


# Turn on debugger by default and reloader
manager.add_command("runserver", Server(
    use_debugger=True,
    use_reloader=True,
    host='0.0.0.0',
    processes=100,
    #    port='9000'
))
#
# manager.add_command("runserver", Server(
#     use_debugger=True,
#     use_reloader=True,
#     host='0.0.0.0',
#     port='5001')
#                     )

manager.add_command('preparevectors', ExtractFeatureVectors())


@manager.option('-i', '--input', dest='input', help='filename')
def books(input):
    from utils import get_all_books
    from bookweb.models import BookStatus
    criteria = {'is_active': True}
    # criteria={'is_active':True}
    get_all_books(input, **criteria)


@manager.option('-o', '--output', dest='output_folder', help='/path/to/output/folder/for/results', required=True)
@manager.option('-s', '--sampling', dest='sampling', help='undersampling method')
def success(output_folder, sampling=None):
    from success import run_success_classification
    # label_file = os.path.join(os.path.dirname(__file__), 'resources', 'success_ratings.txt')
    # label_file = os.path.join(os.path.dirname(__file__), 'resources', 'amazon_books_system.csv')

    features = [  # 'all_imp_features',
        #
        # 'prior', 'most_frequent', 'constant', 'stratified', 'uniform',

        # ['google_word_emb', 'writing_density', 'readability'],
        # ['google_word_emb', 'writing_density', 'categorical_char_ngram_prefix'],
        # ['concepts_score-concepts', 'writing_density'],
        # ['concepts_score-concepts', 'google_word_emb'],
        # ['concepts_score-concepts', 'writing_density', 'categorical_char_ngram_prefix'],
        # ['concepts_score-concepts', 'writing_density', 'categorical_char_ngram_mid_word'],

        # ['categorical_char_ngram_beg_punct',
        #  'categorical_char_ngram_mid_punct',
        #  'categorical_char_ngram_end_punct',
        #  'categorical_char_ngram_multi_word',
        #  'categorical_char_ngram_whole_word',
        #  'categorical_char_ngram_mid_word',
        #  'categorical_char_ngram_space_prefix',
        #  'categorical_char_ngram_space_suffix',
        #  'categorical_char_ngram_prefix',
        #  'categorical_char_ngram_suffix'],
        #
        # ['char_tri', 'char_4_gram', 'char_5_gram']


        ['stress_ngrams', 'stress_scores', 'phonetic', 'phonetic_scores'],

        ['stress_ngrams', 'stress_scores'],

        ['lexicalized', 'gp_lexicalized'],

        ['unlexicalized', 'gp_unlexicalized'],

        ['phr_cls', 'unigram'],
        ['phr_cls', 'bigram'],

        ['stress_ngrams', 'stress_scores', 'writing_density', 'readability'],
        ['writing_density', 'readability'],

        ['char_5_gram', 'unigram'],

        ['categorical_char_ngram_beg_punct', 'unigram'],

        ['categorical_char_ngram_beg_punct', 'bigram'],
        ['char_5_gram', 'bigram'],
        ['unigram', 'bigram']
    ]
    run_success_classification(features, output_folder, sampling)


@manager.option('-o', '--output', dest='output_folder', help='/path/to/output/folder/for/results', required=True)
def success_baseline(output_folder):
    from success import run_success_classification_baseline
    # label_file = os.path.join(os.path.dirname(__file__), 'resources', 'success_ratings.txt')
    # label_file = os.path.join(os.path.dirname(__file__), 'resources', 'amazon_books_system.csv')
    run_success_classification_baseline(output_folder)


@manager.option('-n', '--number', dest='n', help='number of recos', required=True)
@manager.option('-o', '--output', dest='output', help='/path/to/output/file', required=True)
def recommendations(output, n=15):
    import pandas as pd
    from bookweb.engine import Recommendation, FeaturesSimilarity
    from bookweb.models import Book

    def book_title(x):
        # print "Book id: =====  "+x
        book = Book.objects.get(book_id=x)
        return book.title.title()

    # books=['Q9zPCgAAQBAJ','Q1IjBQAAQBAJ','YDyMDAAAQBAJ','dc3pAQAAQBAJ'] #books sent by holly
    books = ['8st8cSkyGt8C', 'VgLKYawnwHgC', 'isbnnotknown']  # Josh books
    features = app.config['FEATURES']
    n = int(n)
    # results={}
    for book in books:
        # most_similar, least_similar = get_n_similar_books(id, 10, features)

        book_obj = Book.objects.get(book_id=book)
        # print book_obj.title
        recommendation = Recommendation()
        feature_layer = FeaturesSimilarity(book_obj, features)
        recommendation.add_layer(feature_layer)

        most_similar, least_similar = recommendation.get_n_similar_books(n, kernel='cosine')
        most_similar['Title'] = most_similar.index.map(book_title)
        most_similar.to_csv(book_obj.title.replace(" ", '_') + '.csv', encoding='utf-8')
        # results[book_obj.title]=similar_books


        # print results
        # df= pd.DataFrame(results)

        # df.to_csv(output, encoding='utf-8')


'''
def separate_training_test():
    #Prepares the training and test data
    #test data consists of books in the test set and their similar BookStatus
    #training set consists of all other active books
    test_books=['Q9zPCgAAQBAJ','Q1IjBQAAQBAJ','YDyMDAAAQBAJ','dc3pAQAAQBAJ']
    train_books=[]
    from booxby.models import Book

    features = app.config['FEATURES']
    n=16
    #results={}
    for book in books:
        # most_similar, least_similar = get_n_similar_books(id, 10, features)
        book_obj=Book.objects.get(book_id=book)
        #print book_obj.title
        recommendation=Recommendation()
        feature_layer=FeaturesSimilarity(book_obj,features)
        recommendation.add_layer(feature_layer)
        most_similar, least_similar = recommendation.get_n_similar_books(n,kernel='cosine')
        similar_books= most_similar.index
        test_books.extends(similar_books)
	test_books=set(test_books)
	print "Total test books: {}".format(len(test_books))
	for book in Book.objects(is_active=True):
		train_books.append(book.book_id)
	print "Total train  books before: {}".format(len(train_books))
	train_books=set(train_books)-test_books
	print "Total train  books after: {}".format(len(train_books))
	return train_books, test_books
@manager.command
def build_and_predict_success_models():
	import joblib
    import pandas as pd
    import os
    from success import build_success_clf_model, get_prediction_for
	features = app.config['FEATURES']
    success_models=app.config['SUCCESS_MODELS']
	train_data, test_data=separate_training_test()
	all_results={}
	for feature in features:
		print "Building model for feature :{}".format(feature)
		model , feature_obj= build_success_clf_model(train_data,feature)
	    dump_file="-".join(feature) if isinstance(feature, list) else feature
        joblib.dump(model,os.path.join(success_models,dump_file+'.success'))

		#predict on the test data
		results= get_prediction_for(test_data, model, feature_obj)
		all_results[feature]=results
		print results
	df=pd.DataFrame(all_results) ##need to check here
	df.to_csv(df)
'''


def separate_training_test():
    '''
		Prepares the training and test data
        test data consists of books in the test set and their similar BookStatus
        training set consists of all other active books
    '''
    # test_books = ['Q9zPCgAAQBAJ', 'Q1IjBQAAQBAJ', 'YDyMDAAAQBAJ', 'dc3pAQAAQBAJ']  # Lisa Jewells books
    test_books = ['3ZH6oAEACAAJ', 'HKCaPgAACAAJ', 'yiIrAQAAIAAJ', 'B00658MH7K']  # Holly's  books
    # test_books = ['8st8cSkyGt8C','VgLKYawnwHgC','isbnnotknown'] #Josh books
    train_books = []

    from bookweb.models import Book
    from bookweb.engine import Recommendation, FeaturesSimilarity

    features = app.config['FEATURES']
    n = 16
    # results={}
    tmp_books = [x for x in test_books]
    for book in test_books:
        # most_similar, least_similar = get_n_similar_books(id, 10, features)

        book_obj = Book.objects.get(book_id=book)
        # print book_obj.title
        recommendation = Recommendation()
        feature_layer = FeaturesSimilarity(book_obj, features)
        recommendation.add_layer(feature_layer)

        most_similar, least_similar = recommendation.get_n_similar_books(n, kernel='cosine')
        similar_books = most_similar.index
        tmp_books.extend(similar_books)

    test_books = set(tmp_books)
    print "Total test books: {} ".format(len(test_books))

    for book in Book.objects(is_active=True):
        train_books.append(book.book_id)

    print "Total train  books before: {}".format(len(train_books))
    train_books = set(train_books) - test_books

    print "Total train  books after: {}".format(len(train_books))
    return train_books, test_books

@manager.command
def findSimilarity0():
    from bookweb.recommendation import findSimilarity0
    findSimilarity0()

@manager.command
def build_and_predict_success_models():
    import joblib

    import pandas as pd
    import os

    from success import build_success_clf_model, get_prediction_for

    features = app.config['FEATURES']
    success_models = app.config['SUCCESS_MODELS']
    train_data, test_data = separate_training_test()
    all_results = {}
    for feature in features:
        print "Building model for feature :{}".format(feature)
        model, feature_obj = build_success_clf_model(train_data, feature)
        dump_file = "-".join(feature) if isinstance(feature, list) else feature
        joblib.dump(model, os.path.join(success_models, dump_file + '.success'))  # predict on the test data
        results = get_prediction_for(test_data, model, feature_obj)
        all_results["-".join(feature) if isinstance(feature, list) else feature] = results
        print results

    joblib.dump(all_results, 'holly.pkl')
    df = pd.DataFrame(all_results)  ##need to check here
    df.to_csv('holly.csv', encoding='utf-8')


@manager.command
def all_success_prediction():
    from success import all_book_success_clf
    features = app.config['FEATURES']
    output = '/home/sjmaharjan/Booxby/all_pred'

    all_book_success_clf(features, output)


@manager.command
def feature_importance_label():
    from experiments.exp_feature_imp import run_feature_importance_label

    features = [  # WR and Readability
        'writing_density',
        'readability',  # WR and Readability
        # 'writing_density',

        #
        # # #Phonetic Features

        'google_word_emb',
        'phonetic',
        #
        'phonetic_scores',

        'phrasal',
        'clausal',
        'phr_cls',
        # # 'google_word_emb',
        'categorical_char_ngram_beg_punct',
        'categorical_char_ngram_mid_punct',
        'categorical_char_ngram_end_punct',
        'categorical_char_ngram_multi_word',
        'categorical_char_ngram_whole_word',
        'categorical_char_ngram_mid_word',
        'categorical_char_ngram_space_prefix',
        'categorical_char_ngram_space_suffix',
        'categorical_char_ngram_prefix',
        'categorical_char_ngram_suffix',

        # 'phonetic_scores',

        # Syntactic Features
        'pos',

        'lexicalized',
        'unlexicalized',
        'gp_lexicalized',
        'gp_unlexicalized',
        #
        #
        # # char ngrams
        'char_tri',
        'char_4_gram',
        'char_5_gram',
        #
        # # Typed char ngrams
        #
        #
        #
        'unigram', 'bigram', 'trigram',
        'stress_ngrams',
        'stress_scores',
        # 'readability'

    ]
    # label_file = os.path.join(os.path.dirname(__file__), 'resources', 'statistics_of_reviews_amazon.tsv')

    for feature in features:
        print ('feature name {}'.format(feature))
        # run_feature_importance_label(label_file, [feature])
        run_feature_importance_label([feature])


@manager.option('-i', '--input', dest='input', help='/path/to/feature/correlation/dir')
def allimpfeatures(input):
    from experiments.exp_feature_imp import prepare_imp_features
    prepare_imp_features(input, n=50)


@manager.command
def feature_importance():
    from experiments.exp_feature_imp import run_feature_importance
    from celery import group
    features = [  # WR and Readability
        # 'writing_density',
        # 'readability',# WR and Readability
        # 'writing_density',

        #
        # # #Phonetic Features

        'google_word_emb',
        'phonetic',
        #
        'phonetic_scores',

        #    'phrasal',
        # 'clausal',
        # 'phr_cls',
        # 'google_word_emb',
        'categorical_char_ngram_beg_punct',
        'categorical_char_ngram_mid_punct',
        'categorical_char_ngram_end_punct',
        'categorical_char_ngram_multi_word',
        'categorical_char_ngram_whole_word',
        'categorical_char_ngram_mid_word',
        'categorical_char_ngram_space_prefix',
        'categorical_char_ngram_space_suffix',
        'categorical_char_ngram_prefix',
        'categorical_char_ngram_suffix',

        # 'phonetic_scores',

        # Syntactic Features
        # 'pos',

        'lexicalized',
        'unlexicalized',
        'gp_lexicalized',
        'gp_unlexicalized',

        # char ngrams
        'char_tri',
        'char_4_gram',
        'char_5_gram',

        # Typed char ngrams



        'unigram', 'bigram', 'trigram',

    ]
    label_file = os.path.join(os.path.dirname(__file__), 'resources', 'statistics_of_reviews_amazon.tsv')

    jobs = group(run_feature_importance.s(label_file, [feature]) for feature in features)

    jobs.apply_async()


@manager.command
def avgsent():
    from success import average_sentences
    # label_file = os.path.join(os.path.dirname(__file__), 'resources', 'amazon_books_system.csv')
    average_sentences()

    # (907, 907)->666.9


@manager.command
def print_total_features():
    '''
    prints the names of the feature and total features
    :return:
    '''
    from bookweb.engine import feature_stats

    feature_stats()


@manager.option('-i', '--input', dest='input', help='/path/to/feature/correlation/dir')
def top_features(input):
    from experiments.exp_feature_imp import rank_features
    features = ['pos', 'phonetic_scores', 'writing_density', 'categorical_char_ngram_space_suffix',
                'categorical_char_ngram_mid_word', 'categorical_char_ngram_multi_word', 'categorical_char_ngram_prefix',
                'categorical_char_ngram_suffix', 'categorical_char_ngram_whole_word',
                'clausal', 'phr', 'phr_cls']
    for feature in features:
        rank_features(input, feature)


@manager.option('-i', '--input', dest='input', help='/path/to/result/folder')
def displayresults(input):
    from utils import display_success_results
    display_success_results(input)


@manager.option('-i', '--input', dest='input', help='/path/to/result/folder')
def collect_all_results(input):
    '''
    interface to collect results
    :param input:
    :return:
    '''
    from utils import collect_results
    from flask import current_app

    order = current_app.config['FEATURES']

    # collect_results(order,input,'clf')
    collect_results(order, input, 'clf')


@manager.option('-i', '--input', dest='input', help='/path/to/result/folder')
def deactivate(input):
    from utils import deactivate
    deactivate(input)



@manager.command
def delteComments0():
    from bookweb.utils import delteComments
    delteComments()

@manager.command
def insertauthorsbooks1():
    from bookweb.utils import save_authors_book
    save_authors_book()


@manager.command
def insertauthorsbooks():
    from bookweb.utils import saveGutenbergAuthor
    saveGutenbergAuthor()

@manager.command
def insertNewGoogleBook():
    from bookweb.utils import save_new_google_book
    save_new_google_book()


@manager.command
def tofile():
    from bookweb.utils import getInfoToFile
    getInfoToFile()

@manager.command
def crawlAmazon():
    from bookweb.utils import crawl
    crawl()

@manager.option('-i', '--input', dest='input_folder', help='/path/to/input/folder/', required=True)
@manager.option('-t', '--type', dest='type', help='{Authors|GoogleBook}')
def insertgooglebooks(input_folder, type):
    print("ooooooooooooooo")
    from bookweb.utils import save_googlebook_to_db
    save_googlebook_to_db(input_folder, type)


@manager.command
def insertgutenbergbooks():
    from bookweb.utils import save_gutenberg_books
    save_gutenberg_books()

@manager.command
def testNew():
    from bookweb.books import sentimentAnalyse
    sentimentAnalyse()

@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def update_book_tags(input_file=None):
    from bookweb.utils import update_book_experiential_language_tags
    update_book_experiential_language_tags(input_file)


@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def write_recommendation_to_file(input_file=None):
    from bookweb.utils import write_all_the_recommendations_to_file
    write_all_the_recommendations_to_file()


@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def tag_vs_genre(input_file=None):
    from bookweb.utils import write_tag_vs_genre_information
    write_tag_vs_genre_information(input_file)


@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def write_similar_books_tags(input_file=None):
    from bookweb.utils import write_similar_books_tags
    write_similar_books_tags()


@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def generate_synthetic_tags(input_file=None):
    from bookweb.utils import generate_synthetic_tags
    generate_synthetic_tags()


@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def predict_tags(input_file=None):
    from Experiential_Tags.Predict_Tag_by_Multilabel_Classification import run_tag_prediction
    run_tag_prediction()


@manager.option('-i', '--input', dest='input_file', help='/path/to/input/folder/')
def meta_tag(input_file=None):
    from Experiential_Tags.MetaLabeler import run_tag_prediction
    run_tag_prediction()


@manager.option('-i', '--input', dest='input_folder', help='/path/to/input/folder/')
def runsenticparser(input_folder=None):
    from parsers.sentic import sentic_parser
    sentic_parser.run_sentic_parser(input_folder)


@manager.option('-i', '--input', dest='input_folder', help='/path/to/input/folder/', required=True)
def runstanfordparser(input_folder):
    from parsers.stanford_parser import run_stanford_parser
    run_stanford_parser(input_folder)


@manager.command
def updateauthorsbook():
    from bookweb.utils import upate_authors_book
    upate_authors_book()


@manager.command
def updategooglebook():
    from bookweb.utils import update_google_book
    update_google_book()


@manager.option('-i', '--input', dest='input', help='/path/to/input/folder/')
@manager.option('-o', '--output', dest='output', help='/path/to/output/folder/')
def extract_content(input, output):
    # run st parser and sentic extractor
    from bookweb.admin import get_book_content
    for book in os.listdir(input):
        with codecs.open(os.path.join(output, book.rsplit('.', 1)[0] + '.txt'), 'w', encoding='utf-8') as f_out:
            content = get_book_content(os.path.join(input, book))
            try:
                f_out.write(content)
                f_out.flush()
            except UnicodeDecodeError as u:
                content = content.decode('latin1')
                f_out.write(content)
                f_out.flush()
    print ('Done extracting text form pdf/doc/')


@manager.option('-i', '--input', dest='input', help='/path/to/input/folder/')
def parse_authors_books(input):
    # run st parser and sentic extractor
    from parsers.tasks import run_stanford_parser, run_sentic_parser
    from celery import group
    jobs = group((run_stanford_parser.s(os.path.join(input, book)) | run_sentic_parser.s())
                 for book in os.listdir(input))

    jobs.apply_async()


@manager.option('-o', '--output', dest='output_folder', help='/path/to/output/folder/', required=True)
def dumpcontent(output_folder):
    from bookweb.utils import dump_data
    dump_data(output_folder)


@manager.option('-n', '--number', dest='n', help='number of top books', required=True)
@manager.option('-o', '--output', dest='filename', help='/path/to/output/file/', required=True)
def highvaluebooks(n, filename):
    from bookweb.engine import get_top_n_vectors
    features = ['writing_density',
                'readability',

                # #Phonetic Features
                'phonetic',

                'phonetic_scores',
                # Google w2v embedding
                'google_word_emb',
                # sentic concepts and scores
                ['concepts_score', 'concepts']]
    get_top_n_vectors(features, 10, filename)


@manager.option('-o', '--output', dest='filename', help='/path/to/output/file/', required=True)
@manager.option('-f', '--feature', dest='feature', help='feature name', required=True)
def dumpfeaturevalues(filename, feature):
    from bookweb import fvs
    from bookweb.models import Book
    import pandas as pd
    import scipy.sparse as sp

    if isinstance(feature, list):
        feature = "-".join(feature)
    books = fvs.feature_vectors[feature].ids()
    vectors = fvs.feature_vectors[feature].vectors()
    if sp.issparse(vectors):
        vectors = vectors.toarray()
    model = fvs.feature_vectors[feature].model()
    feature_names = model.get_feature_names()

    feature_data = []

    for book in Book.objects(is_active=True):
        book_idx = books.index(book.book_id)
        row = {}
        row['book_id'] = book.book_id
        row['book_title'] = book.title.title()
        row['book_isbn_10'] = book.isbn_10
        row['book_isbn_13'] = book.isbn_13

        for f_name, f_value in zip(feature_names, vectors[book_idx]):
            if feature.startswith('concepts_score'):
                if f_name in ['concepts_score__avg_sensitivity', 'concepts_score__avg_attention',
                              'concepts_score__avg_pleasantness', 'concepts_score__avg_aptitude',
                              'concepts_score__avg_polarity']:

                    row[f_name] = f_value
                else:
                    row[f_name] = f_value
        feature_data.append(row)

    df = pd.DataFrame(feature_data)
    df.to_csv(filename, sep='\t', encoding='utf-8', index=False)


@manager.option('-o', '--output', dest='output_folder', help='/path/to/output/folder/', required=True)
def downloadgooglebookinfo(output_folder):
    from scraper.google_books import download_meta_info
    dirname = os.path.dirname(__file__)
    with codecs.open(os.path.join(dirname, 'resources', 'books.yaml'), 'r', encoding='utf-8') as f:
        books_config = yaml.load(f)

    for book in books_config['books']:
        data = download_meta_info(book['google_book_id'])
        out_fname = os.path.join(output_folder, book['google_book_id'] + '_' + str(book['isbn10']) + '.json')
        with codecs.open(out_fname, mode='w', encoding='utf-8') as out:
            json.dump(data, out)


@manager.option('-i', '--input', dest='input_dir', help='/path/to/input/folder/', required=True)
def import_data(input_dir):
    import pandas as pd
    from bookweb.utils import save_to_db
    meta_info = os.path.join(input_dir, 'SkywriterRX_{folder}.csv.xlsx'.format(folder=os.path.basename(input_dir)))
    print ("Loading meta info from %s" % meta_info)

    df = pd.read_excel(meta_info, index_col=0, converters={'ISBN_13': str, 'ISBN_10': str, 'PublicationDate': str})
    df.dropna(how='all', inplace=True)
    df.fillna('', inplace=True)
    for book in os.listdir(os.path.join(input_dir, 'Content_Single_File_Text')):
        book_fname = os.path.join(input_dir, 'Content_Single_File_Text', book)
        google_book_id, isbn_13 = book.rsplit('_', 1)
        try:
            save_to_db(book_fname, google_book_id, df.loc[google_book_id])
            print ('Saved book %s' % book)
        except Exception as ex:
            print (ex)
            print ("---->No meta data for %s" % book_fname)


@manager.option('-i', '--input', dest='input_dir', help='/path/to/input/folder/', required=True)
def learningcurve(input_dir):
    from learning_curve import learning_curve
    from celery import group
    features = [
        'unigram', 'bigram', 'trigram',

        # char ngrams
        'char_tri',
        'char_4_gram',
        'char_5_gram',

        # Typed char ngrams
        'categorical_char_ngram_beg_punct',
        'categorical_char_ngram_mid_punct',
        'categorical_char_ngram_end_punct',
        'categorical_char_ngram_multi_word',
        'categorical_char_ngram_whole_word',
        'categorical_char_ngram_mid_word',
        'categorical_char_ngram_space_prefix',
        'categorical_char_ngram_space_suffix',
        'categorical_char_ngram_prefix',
        'categorical_char_ngram_suffix',

        # Syntactic Features
        'pos',
        # 'phrasal',
        # 'clausal',
        # 'phr_cls',
        'lexicalized',
        'unlexicalized',
        'gp_lexicalized',
        'gp_unlexicalized',

        # WR and Readability
        'writing_density',
        'readability',
        #
        # # #Phonetic Features
        'phonetic',
        #
        'phonetic_scores',
        # Google w2v embedding
        # ['concepts_score','concepts']
    ]
    jobs = group(learning_curve.s(input_dir, feature) for feature in features)

    jobs.apply_async()


@manager.option('-i', '--input', dest='input_dir', help='/path/to/input/folder/', required=True)
def gen_vectors(input_dir):
    from dump_vectors import generate_vectors
    features = [
        # 'writing_density',
        # 'readability',
        #  'phonetic_scores',
        'concepts_score']
    generate_vectors(input_dir, features)


########################################################################################################################


@manager.command
def experiment_kernels():
    from experiments import exp_kernels
    exp_kernels.recommendations(n=20)


@manager.command
def test():
    from dump_vectors import test
    test()





@manager.command
def test_stress_features():
    from dump_vectors import BookDataWrapper
    from features.phonetic import StressNgramsVectorizer, StressFeatures

    # text="""What a lark! What a plunge! For so it had always seemed to her, when, with a little squeak of the hinges,
    # which she could hear now, she had burst open the French windows and plunged at Bourton into the open air.
    # How fresh, how clam, stiller than this of course, the air was in the early morning; like the flap of a wave, the kiss of a wave, chill and sharp and yet
    # (for a girl of eighteen as she then was) solemn, feeling as she did,  standing there at the open window, that something awful was about to happen.
    # """

    text = "What a lark! What a plunge! For so it had always seemed to her."

    # print (get_stress_markers(text))

    book = BookDataWrapper(book_id="b1", isbn_10="22222", content=text)
    sf = StressFeatures()
    print(sf.fit_transform([book]))

    sngrams = StressNgramsVectorizer(ngram_range=(3, 3))
    print (sngrams.fit_transform([book]))
    print (sngrams.get_feature_names())


if __name__ == "__main__":
    # insertgutenbergbooks()
    manager.run()
