__author__ = 'suraj'

import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    BOOKS_PER_PAGE =25
      #CELERY
    CELERY_BROKER_URL = "amqp://ritual:books@localhost:5672/ritual"
    CELERY_BACKEND_URL = "amqp://ritual:books@localhost:5672/ritual"
    CELERY_ACCEPT_CONTENT = ['pickle','json']
    CELERY_IMPORTS=("parsers.tasks","dump_vectors","bookweb.admin")
    STANFORD_PARSER='./stanford-parser-full-2016-10-31/lexparser.sh' #download stanford parser and then point to lexparser.sh
    UPLOAD_FOLDER= os.path.join(basedir,'uploads')
    PDF_PARSE=os.path.join(basedir,'pdf')



    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    #DEBUG_TB_PANELS = ['flask.ext.mongoengine.panels.MongoDebugPanel','flask_debugtoolbar.panels.timer.TimerDebugPanel','flask_debugtoolbar.panels.route_list.RouteListDebugPanel','flask_debugtoolbar.panels.request_vars.RequestVarsDebugPanel','flask_debugtoolbar.panels.config_vars.ConfigVarsDebugPanel']


    MONGODB_SETTINGS = {'DB': "Books", 'host':"127.0.0.1"}
    STANFORD_PARSER_OUTPUT = os.path.join(basedir,'stanford_parse')
    SENTIC_PARSE_OUTPUT=os.path.join(basedir,'sentic_parse')
    #GOOGLE_BOOK_INFO='/home/suraj/resouces/booxby/google_book'
    ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx'])
    UPLOAD_FOLDER= os.path.join(basedir,'uploads')
    TEMP=os.path.join(basedir,'tmp')



    FEATURES = [

        # Lexical Features
        # word ngrams

        'unigram',
        'bigram',
       #  'trigram',
       #
        ['unigram', 'bigram'],
      #  ['bigram', 'trigram'],
      #  ['unigram', 'bigram', 'trigram'],

        # char ngrams
       'char_tri',
       'char_4_gram',
       'char_5_gram',
       ['char_tri', 'char_4_gram', 'char_5_gram'],

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
        'phrasal',
        'clausal',
        'phr_cls',
        'lexicalized',
        'unlexicalized',
        'gp_lexicalized',
        'gp_unlexicalized',

        # WR and Readability
        'writing_density',
        'readability',

       # #Phonetic Features
        #'phonetic',
       
        #'phonetic_scores',
        # Google w2v embedding
        # 'google_word_emb',
        #sentic concepts and scores
       # ['concepts_score','concepts']

    ]

    FEATURES_IN_USE = [

           # Lexical Features
        # word ngrams

        'unigram',
        # 'bigram',
        # 'trigram',
        ['bigram', 'trigram'],
       #  ['unigram', 'bigram', 'trigram'],
       #
       #  # char ngrams
       #  'char_tri',
       #  'char_4_gram',
       #  'char_5_gram',
       #  ['char_tri', 'char_4_gram', 'char_5_gram'],
       #
       #  # Typed char ngrams
       #  'categorical_char_ngram_beg_punct',
       #  'categorical_char_ngram_mid_punct',
       #  'categorical_char_ngram_end_punct',
       #  'categorical_char_ngram_multi_word',
       #  'categorical_char_ngram_whole_word',
       #  'categorical_char_ngram_mid_word',
       #  'categorical_char_ngram_space_prefix',
       #  'categorical_char_ngram_space_suffix',
       #  'categorical_char_ngram_prefix',
       #  'categorical_char_ngram_suffix',
       #
       #  # Syntactic Features
       #  'pos',
       #  'phrasal',
       #  'clausal',
       #  'phr_cls',
       #  'lexicalized',
       #  # 'unlexicalized',
       #  'gp_lexicalized',
       #  'gp_unlexicalized',
       #
       #  # WR and Readability
       #  'writing_density',
       #  'readability',
       #
       #
       # # #Phonetic Features
       #  'phonetic',
       #  'phonetic_scores',
       #  # Google w2v embedding
       #  'google_word_emb',
       #  #sentic concepts and scores
       #   ['concepts_score','concepts']

        'google_word_emb',
       # 'concepts_score-concepts',
       # ['google_word_emb', 'writing_density', 'readability'],
       # ['google_word_emb', 'writing_density', 'categorical_char_ngram_prefix'],
       # ['concepts_score-concepts','writing_density'],
       # ['concepts_score-concepts','google_word_emb'],
       # ['concepts_score-concepts', 'writing_density', 'categorical_char_ngram_prefix']
    ]
    SECRET_KEY = 'eyJhbGciOiJIUzI1NiIsImV4cCI6MTc3MjAxMjcxMCwiaWF0IjoxNDYxMDA4NzEwfQ.eyJjb25maXJtIjoyM30.tV259JmanhOv29Jqzbr_zKsoJXIFPN0Z-OrNrb-UF7k'  # Serializer("BooxbySecretKey", expires_in = 311004000)

    VECTORS = os.path.join(basedir,'vectors')
    STAGGED_VECTORS =os.path.join(basedir,'stagged_vectors')
    IGNORE = []

    DATA_DIR=''





class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    pass


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
