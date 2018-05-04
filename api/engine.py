
__author__ = 'suraj'



from sklearn.metrics.pairwise import cosine_similarity
from bookweb import fvs
import pandas as pd
from flask import current_app
from bookweb.models import Book
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

def change_nan_to_num(Y):
   print ("engine start")
   X = coo_matrix(Y)
   for i, j, v in zip(X.row, X.col, X.data):
       if (np.isnan(v) or np.isinf(v)):
           print("isnan", i, j, v)
           Y[i,j] = 0.
   print ("engine end"	)

def similarity(book_index, X):
    """ Computes the cosine similarity of a vecto with all other vectors

    Parameters
    ------------
    book_index : int
        index of the book vector in X

    X: numpy 2D array
         document vectors in row

    Returns
    --------
    similarity scores : numpy array with cosine similarity scores with all other vectors


    """
    #return cosine_similarity(X[book_index], X)
   # X[book_index]
    #print X.shape
    #print type(X)
    #print "engine 1", X.shape
    #X=np.nan_to_num(X)
   # X[np.isnan(X)] = 0.
   # print "engine 3", X.shape
   # X[book_index]
    #print "engine x2"
    # if sp.issparse(X):
    #     change_nan_to_num(X)
    # else:
    #     X=np.nan_to_num(X)
    if np.count_nonzero(X[book_index]) > 0:
        return cosine_similarity(X[book_index], X)

    else:
        sim=np.zeros((1,X.shape[0]), dtype=np.float32)
        for i,book_vector in enumerate(X):
            if np.count_nonzero(book_vector)>0:
                sim[1,i]=cosine_similarity(X[book_index], book_vector)
            else:
                sim[1,i]=0
        return sim





def get_n_similar_books(book_id,n,features):
    """ Computes the cosine similarity of provied book with other books in collection
        with provieds features

        Then produce a single score by averaging all the cosine score

    Parameters
    -----------
    book_id : book unique identifier {google book id or gutenberg id}

    n : int
        number of similar and dissimilar books to return

    features : list of features
        The features to considers for computing the cosine similarity


    Returns:
    n_similar and n_dissimilar books


    """

    dfs=[]
    if n<=0:
        n=10
    if not features:
        features=current_app._get_current_object().config['FEATURES']
    ignore_book_lst=[ book.book_id for book in Book.objects(is_active=False)]

    for feature in features:
        if isinstance(feature,list):
            feature="-".join(feature)
        book_index= fvs.feature_vectors[feature].ids().index(book_id)
        # print book_index
        print "Feature ............",feature
        cosine_values= similarity(book_index,fvs.feature_vectors[feature].vectors())
        data={feature : pd.Series(cosine_values[0], index=fvs.feature_vectors[feature].ids())}
        df = pd.DataFrame(data)
        df.drop(ignore_book_lst, inplace=True,errors='ignore')
        dfs.append(df)

    result_df=pd.concat(dfs, axis=1, join='inner')

    app = current_app._get_current_object()

    if ignore_book_lst:
        result_df.drop(ignore_book_lst, inplace=True,errors='ignore')
    if app.config['IGNORE']:
        result_df.drop(app.config['IGNORE'], inplace=True)

    result_df['avg']=result_df.mean(axis=1)

    return result_df.nlargest(n,'avg'), result_df.nsmallest(n,'avg')



def get_feature_values(feature_name,book_id):
    book_index = fvs.feature_vectors[feature_name].ids().index(book_id)
    model= fvs.feature_vectors[feature_name].model()
    feature_names=model.get_feature_names()
    vector=fvs.feature_vectors[feature_name].vectors()

    if sp.issparse(vector):
        book_vector=np.ravel(vector[book_index].toarray())
    else:
        book_vector=np.ravel(vector[book_index])
    return zip(feature_names,book_vector)




class RecommendationComponent(object):
    """Abstract class"""

    def __init__(self, *args, **kwargs):
        self.ignore_book_lst = [book.book_id for book in Book.objects(is_active=False)]

    def get_result_col_name(self):
        pass

    def compute_similarity(self):
        pass


class SimilarityLayer(RecommendationComponent):
    pass


class Composite(RecommendationComponent):
    """interface class and maintains the tree recursive structure"""

    def __init__(self, *args, **kwargs):
        RecommendationComponent.__init__(self, *args, **kwargs)

        # similarity layers
        self.children = []

    def append_child(self, child):
        """Method to add a similarity criteria"""
        self.children.append(child)

    def remove_child(self, child):
        """Method to remove similarity criteria"""
        self.children.remove(child)




class FeaturesSimilarity(SimilarityLayer):
    def __init__(self, book,features=None, *args, **kwargs):
        SimilarityLayer.__init__(self, *args, **kwargs)
        self.book=book
        self.features = features if features else current_app._get_current_object().config['FEATURES']

    def get_result_col_name(self):
        return "feature_avg"

    def compute_similarity(self):
        dfs = []
        for feature in self.features:
            if isinstance(feature, list):
                feature = "-".join(feature)
            book_index = fvs.feature_vectors[feature].ids().index(self.book.book_id)
            # print book_index
            cosine_values = similarity(book_index, fvs.feature_vectors[feature].vectors())
            data = {feature: pd.Series(cosine_values[0], index=fvs.feature_vectors[feature].ids())}
            df = pd.DataFrame(data)
            df.drop(self.ignore_book_lst, inplace=True, errors='ignore')
            dfs.append(df)

        result_df = pd.concat(dfs, axis=1, join='inner')

        app = current_app._get_current_object()

        if self.ignore_book_lst:
            result_df.drop(self.ignore_book_lst, inplace=True, errors='ignore')
        if app.config['IGNORE']:
            result_df.drop(app.config['IGNORE'], inplace=True)

        result_df[self.get_result_col_name()] = result_df.mean(axis=1)

        return result_df




class ExperientialLanguageSimilarity(SimilarityLayer):
    def __init__(self, book, language_tags, *args, **kwargs):
        SimilarityLayer.__init__(self, *args, **kwargs)
        self.book = book
        self.language_tags = language_tags

    def get_result_col_name(self):
        return "explang_avg"


    def transform(self,book):
        if book.has_experiential_languages():
            exp_lang_dic =book.get_experiential_languages()
            emb,weight=np.zeros((len(exp_lang_dic.keys()),explangemb.embeddings.get_emb_size()),np.float32),[]
            for i,(tag, freq) in enumerate(exp_lang_dic.items()):
                emb[i]=explangemb.embeddings.phrase_embedding(tag)
                weight.append(freq)

            return emb,np.array(weight)
        else: #return zero vector if no exp lang tags for book and weight is zero
            return None,None


    def compute_similarity(self):

        dfs = []
        source_emb,source_weights=self.transform(self.book)

        for bk in Book.objects( is_active=True):
            row={}
            tag_emb,tag_weights=self.transform(bk)
            row['id']=bk.book_id
            if tag_emb is not None:
            # TODO: dependency injetion of similatiry funtion later, now it uses cosine similarity
                similarity_mat=cosine_similarity(source_emb,tag_emb)
                row[self.get_result_col_name()]=np.average(np.ravel(similarity_mat),weights=[x*y for x in source_weights for y in tag_weights])
            else:
                row[self.get_result_col_name()]=0.0
            dfs.append(row)

        result_df=pd.DataFrame(dfs)
        result_df.set_index('id', inplace=True)
        return result_df



class Aggreagator(Composite):
    def __init__(self, *args, **kwargs):
        Composite.__init__(self, *args, **kwargs)

    def get_result_col_name(self):
        return "avg"


    def compute_similarity(self):
        first=True
        for layer in self.children:
            print (layer.get_result_col_name())
            if first:
                result_df=layer.compute_similarity()
                first=False
            else:

                result_df=result_df.join(layer.compute_similarity(),how='inner')


        # print(result_df.columns)
        result_df[self.get_result_col_name()]=result_df[ [layer.get_result_col_name() for layer in self.children]].mean(axis=1)
        return result_df



class Recommendation(object):
     def __init__(self, *args, **kwargs):
         self.model=Aggreagator()

     def add_layer(self,layer):
         self.model.append_child(layer)

     def get_n_similar_books(self, n):
        result_df=self.model.compute_similarity()
        return result_df.nlargest(n, self.model.get_result_col_name()), result_df.nsmallest(n, self.model.get_result_col_name())


