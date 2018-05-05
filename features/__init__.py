__author__ = 'suraj'

from sklearn.pipeline import FeatureUnion
from . import lexical
from . import syntactic
from . import embeddings
from . import phonetic
from . import readability
from . import sentiments
from . import writing_density
from . import dumped_features
import nltk
import string

__all__ = ['lexical', 'embeddings', 'phonetic', 'readability', 'writing_density', 'sentiments', 'get_feature',
           'create_feature', 'dumped_features']

def preprocess(x):
    return x.replace('\n', ' ').replace('\r', '').replace('\x0C', '').lower()


def get_feature(f_name):
    """Factory to create features objects

    Parameters
    ----------
    f_name : features name

    Returns
    ----------
    features: BaseEstimator
        feture object

    """
    features_dic = dict(
        unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), preprocessor=preprocess,tokenizer=nltk.word_tokenize, analyzer="word",
                                             lowercase=True, min_df=2),
        bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2),  preprocessor=preprocess,tokenizer=nltk.word_tokenize, analyzer="word",
                                            lowercase=True, min_df=2),
        trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3),  preprocessor=preprocess,tokenizer=nltk.word_tokenize, analyzer="word",
                                             lowercase=True, min_df=2),

        #
        # #char ngram
        char_tri=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), preprocessor=preprocess,analyzer="char",
                                              lowercase=True, min_df=2),
        char_4_gram=lexical.NGramTfidfVectorizer(ngram_range=(4, 4),preprocessor=preprocess, analyzer="char", lowercase=True, min_df=2),

        char_5_gram=lexical.NGramTfidfVectorizer(ngram_range=(5, 5),preprocessor=preprocess, analyzer="char", lowercase=True, min_df=2),

        # categorical character ngrams
        categorical_char_ngram_beg_punct=lexical.CategoricalCharNgramsVectorizer(beg_punct=True,preprocessor=preprocess, ngram_range=(3, 3)),
        categorical_char_ngram_mid_punct=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, mid_punct=True),
        categorical_char_ngram_end_punct=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, end_punct=True),

        categorical_char_ngram_multi_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, multi_word=True),
        categorical_char_ngram_whole_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, whole_word=True),
        categorical_char_ngram_mid_word=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, mid_word=True),

        categorical_char_ngram_space_prefix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess,
                                                                                    space_prefix=True),
        categorical_char_ngram_space_suffix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),
                                                                                    space_suffix=True),

        categorical_char_ngram_prefix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3), preprocessor=preprocess,prefix=True),
        categorical_char_ngram_suffix=lexical.CategoricalCharNgramsVectorizer(ngram_range=(3, 3),preprocessor=preprocess, suffix=True),

        # #skip gram
        two_skip_3_grams=lexical.KSkipNgramsVectorizer(k=2, ngram=3,preprocessor=preprocess, lowercase=True),
        two_skip_2_grams=lexical.KSkipNgramsVectorizer(k=2, ngram=2,preprocessor=preprocess, lowercase=True),

        # pos
        pos=syntactic.POSTags(ngram_range=(1, 1), tokenizer=string.split, analyzer="word", use_idf=False, norm='l1'),

        # #phrasal and clausal
        phrasal=syntactic.Constituents(PHR=True),
        clausal=syntactic.Constituents(CLS=True),
        phr_cls=syntactic.Constituents(PHR=True, CLS=True),

        # #lexicalized and unlexicalized production rules
        lexicalized=syntactic.LexicalizedProduction(use_idf=False),
        unlexicalized=syntactic.UnLexicalizedProduction(use_idf=False),
        gp_lexicalized=syntactic.GrandParentLexicalizedProduction(use_idf=False),
        gp_unlexicalized=syntactic.GrandParentUnLexicalizedProduction(use_idf=False),

        # writing density
        writing_density=writing_density.WritingDensityFeatures(),

        # readability
        readability=readability.ReadabilityIndicesFeatures(),

        concepts=sentiments.SenticConceptsTfidfVectorizer(ngram_range=(1, 1), tokenizer=string.split, analyzer="word",
                                                          lowercase=True, binary=True, use_idf=False),

        concepts_score=sentiments.SenticConceptsScores(),

        google_word_emb=embeddings.Word2VecFeatures(tokenizer=nltk.word_tokenize, analyzer="word",
                                                    lowercase=True,
                                                    model_name='/home/suraj/resouces/GoogleNews-vectors-negative300.bin.gz'),

        # phonetics
        phonetic=phonetic.PhoneticCharNgramsVectorizer(ngram_range=(3, 3), analyzer='char', min_df=2, lowercase=False),

        phonetic_scores=phonetic.PhonemeGroupBasedFeatures(),



    )

    return features_dic[f_name]


def create_feature(feature_names):
    """Utility function to create features object

    Parameters
    -----------
    feature_names : features name or list of features names


    Returns
    --------
    a tuple of (feature_name, features object)
       lst features names are joined by -
       features object is the union of all features in the lst

    """
    try:
        if isinstance(feature_names, list):
            return ("-".join(feature_names), FeatureUnion([(f, get_feature(f)) for f in feature_names]))
        else:

            return (feature_names, get_feature(feature_names))
    except Exception as e:
        raise ValueError('Error in function ')
