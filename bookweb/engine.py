__author__ = 'suraj'

from sklearn.metrics.pairwise import  PAIRWISE_KERNEL_FUNCTIONS
from sklearn.metrics.pairwise import cosine_similarity
from bookweb import fvs, explangemb
import pandas as pd
from flask import current_app
from bookweb.models import Book
from numpy import linalg
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix


def change_nan_to_num(Y):
	X = coo_matrix(Y)
	for i, j, v in zip(X.row, X.col, X.data):
		if (np.isnan(v) or np.isinf(v)):
			print("isnan", i, j, v)
			Y[i,j] = 0.



def similarity(book_index, X,kernel='cosine'):
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
	return PAIRWISE_KERNEL_FUNCTIONS[kernel](X[book_index], X)





def feature_stats():
	features = current_app._get_current_object().config['FEATURES']
	for feature in features:

		if isinstance(feature, list):
			feature = "-".join(feature)
		features_names=fvs.feature_vectors[feature].model().get_feature_names()
		vec=fvs.feature_vectors[feature].vectors()
		print "{}".format(vec.shape)
		print "{},{}".format(feature,len(features_names))



def get_features(book_id, features):
	"""

	:param book_id: book unique identifier {google book id or gutenberg id}
	:param features: list of features


	:return:
	"""

	feature_dict = {}

	if not features:
		features = current_app._get_current_object().config['FEATURES']

		for feature in features:
			if isinstance(feature, list):
				feature = "-".join(feature)
			book_index = fvs.feature_vectors[feature].ids().index(book_id)

			# print(feature)
			# print(book_index)
			feature_dict[feature] = fvs.feature_vectors[feature].vectors()[book_index]

	return feature_dict

def get_feature_values(feature_name, book_id):
	book_index = fvs.feature_vectors[feature_name].ids().index(book_id)
	model = fvs.feature_vectors[feature_name].model()
	feature_names = model.get_feature_names()
	vector = fvs.feature_vectors[feature_name].vectors()

	if sp.issparse(vector):
		book_vector = np.ravel(vector[book_index].toarray())
	else:
		book_vector = np.ravel(vector[book_index])
	return zip(feature_names, book_vector)




def get_top_n_vectors(features,n,filename):
	dfs=[]

	for feature in features:
		feature_df=[]
		if isinstance(feature, list):
			feature = "-".join(feature)
		books = fvs.feature_vectors[feature].ids()
		feature_vectors = fvs.feature_vectors[feature].vectors()
		sparse=False
		if sp.issparse(feature_vectors):
			sparse=True
		print("Loaded feature %s"%feature)
		for book in Book.objects(is_active=True):
			book_idx=books.index(book.book_id)
			if sparse:
				norm=linalg.norm(feature_vectors[book_idx].toarray(),2)
			else:
				norm=linalg.norm(feature_vectors[book_idx],2)
			feature_df.append({'book_id':book.book_id, 'book_title': book.title.title(), 'book_isbn_10': book.isbn_10,
					  'book_isbn_13': book.isbn_13,'norm':norm,"":feature})

		df=pd.DataFrame(feature_df)
		largest=df.nlargest(n,"norm")
		smallest=df.nsmallest(n,"norm")
		dfs.append(pd.concat([largest,smallest]))




		# l2_norms=linalg.norm(feature_vectors,axis=1)
		#
		# norms_array = l2_norms.ravel()
		# top_n = np.argsort(norms_array)[-n:]
		# least_n = np.argsort(norms_array)[:n]
		# interesting_norms = np.hstack([top_n, least_n])
		# book_ids=books[interesting_norms]
		# book_norms=norms_array[interesting_norms]
		# df=pd.DataFrame({'book_id':book_ids.tolist(),'l2norm':book_norms.tolist()})
		# df['feature']=feature
		# df['title']=df['book_id'].apply(lambda x: Book.objects.get(book_id=x).title.title())
		# df['isbn_10']=df['book_id'].apply(lambda x: Book.objects.get(book_id=x).isbn_10)
		# df['isbn_13']=df['book_id'].apply(lambda x: Book.objects.get(book_id=x).isbn_13)




	final_df=pd.concat(dfs)
	final_df.to_csv(filename, sep='\t', encoding='utf-8',index=False)






#######################################################################################################################


class RecommendationComponent(object):
	"""Abstract class"""

	def __init__(self, *args, **kwargs):
		self.ignore_book_lst = [book.book_id for book in Book.objects(is_active=False)]

	def get_result_col_name(self):
		pass

	def compute_similarity(self,kernel='cosine'):
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

	def compute_similarity(self, kernel='cosine'):
		
		dfs = []
		result_df = None

		for feature in self.features:
			
			
			if isinstance(feature, list):
				feature = "-".join(feature)
			app = current_app._get_current_object()
			bookid_ = self.book.book_id
			ids_ = fvs.feature_vectors[feature].ids()
			book_index = -1
			
			try:
				book_index = ids_.index(bookid_)
			
			except ValueError: # if the index is not present...
				book_index = -1
				print("Book %s not found in the result set" % self.book)
					
			if book_index != -1:
				cosine_values = similarity(book_index, app.myFeature[feature],kernel=kernel)
				data = {feature: pd.Series(cosine_values[0], index=ids_)}
				df = pd.DataFrame(data)
				df.drop(self.ignore_book_lst, inplace=True, errors='ignore')
				
				dfs.append(df)

		# we have an issue here as there is a possibility that the text files are not present in the database...

		if len(dfs) > 0:
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


	def compute_similarity(self,kernel='cosine'):

		dfs = []
		source_emb,source_weights=self.transform(self.book)

		for bk in Book.objects( is_active=True):
			row={}
			tag_emb,tag_weights=self.transform(bk)
			row['id']=bk.book_id
			if tag_emb is not None:
				similarity_mat=PAIRWISE_KERNEL_FUNCTIONS[kernel](source_emb,tag_emb)
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

	def compute_similarity(self,kernel='cosine'):

		first=True
		result_df = None
		
		for layer in self.children:
			
			if first:
				result_df=layer.compute_similarity(kernel)
				first=False
			else:
				result_df=result_df.join(layer.compute_similarity(kernel),how='inner')

		result_df[self.get_result_col_name()] = result_df[ [layer.get_result_col_name() for layer in self.children]].mean(axis=1)
		
		return result_df

class Recommendation(object):
	
	def __init__(self, *args, **kwargs):
		self.model=Aggreagator()

	def add_layer(self,layer):
		self.model.append_child(layer)

	def get_n_similar_books(self, n, kernel='cosine'):
		result_df=self.model.compute_similarity(kernel)
		return result_df.nlargest(n, self.model.get_result_col_name()), result_df.nsmallest(n, self.model.get_result_col_name())
