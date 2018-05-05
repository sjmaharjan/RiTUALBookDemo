__author__ = 'suraj'

import codecs

import matplotlib.pyplot as plt
import pickle
import time
from plotly.offline import plot
from markupsafe import Markup
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
import math
from flask import Blueprint, request, redirect, render_template, url_for, flash,jsonify
from flask.views import MethodView
from bookweb.engine import  Recommendation, FeaturesSimilarity, ExperientialLanguageSimilarity
from bookweb.models import Book, GutenbergBook, GoogleBook
from flask import current_app
from flask_wtf import Form
from flask_mongoengine.wtf.orm import model_form
from wtforms import StringField, BooleanField, IntegerField, SelectMultipleField, RadioField, SelectField, HiddenField
from wtforms.validators import DataRequired
import pandas as pd
from pandas import DataFrame
from bookweb.models import RecommendationComment
from bookweb.utils import get_comments_count
from functools import partial
from helpers import dictionary
from collections import defaultdict
from bookweb.books import DetailView
from ast import literal_eval

recommendations = Blueprint('recommendations', __name__, template_folder='templates')
features_view = Blueprint('features', __name__, template_folder='templates')


class FeatureForm(Form):

    #test=SelectMultipleField('lexical',choices=[('unigram','Word unigram'),('bigram','Word bigram'),('unigram__bigram','Word unigram bigram'),('char_tri','char tri'),
                                               # ('char_4_gram','char 4 gra'),('char_5_gram','char 5 gram'),('char_tri__char_4_gram__char_5_gram','char tri, char 4 gram, char 5 gram')])
    char_ngram= BooleanField('Char ngram', default=True)


    # Typed char ngrams
    typed_char_ngrams=BooleanField('Typed char ngram', default=True)

    # Syntactic Features
    syntactic_features=BooleanField('Syntactic Features', default=True)
    # WR and Readability
    writing_density = BooleanField('writing density', default=True)
    readability = BooleanField('readability', default=True)
    #google_word_emb = BooleanField('Word Embeddings', default=True)
    #phonetic = BooleanField('Phonetic Ngram', default=True)
    #phonetic_scores = BooleanField('Phonetic Scores', default=True)
    #concepts_score__concepts = BooleanField('Sentic Concepts', default=True)
    exp_lang_tags = SelectMultipleField('Experiential Language Tags', choices=[])

    reco = IntegerField('Recommendations', validators=[DataRequired()], default=10)


def build_url(x):

    book = Book.objects.get(book_id=x)

    return '<a href="' + url_for('books.detail', id=book.book_id, slug=book.slug) + '">' + book.title.title() + '</a>'

def build_img(x):
    book = Book.objects.get(book_id=x)

    return book.book_id+"_"+book.slug



def my_book_slug(x):
    book = Book.objects.get(book_id=x)

    return book.slug


def myChart(most_similar,index0):


    y=[]
    x=[]
    result=[]
    i=0
    j=0

    for col in most_similar.columns:
        if col != "Title" and col != "avg" and col != "explan_avg" and col != "feature_avg" and col!="image_title" and col!="slug":
            x.append(col)
    for index, row in most_similar.iterrows():
        # print index
        for col in most_similar.columns:
            if  index==index0 and col!="Title" and col!="avg" and col!="explan_avg" and col!="feature_avg" and col!="image_title" and col!="slug":
                y.append(row[col])

        f=go.Bar(
                x=x,
                y=y)
    # print x
    # print "------------"
    # print y
    # print "++++++++++"
    return f

def getAvg(most_similar,index0):
    y=[]
    for col in most_similar.columns:
        print (col)
    for index, row in most_similar.iterrows():
        for col in most_similar.columns:
            if  index==index0 and (col=="avg" or col=="explan_avg" or col=="feature_avg"):
                y.append(math.floor(row[col] * 100) / 100)
    return y







# def generate_url(feature_name, src_book,des_book,x):
#     return '<a href="' + url_for('features.list', feature_name=feature_name, src_id=src_book,des_id=des_book) + '">' + x + '</a>'
#         # return '<a href="%s"> %s </a>' % (
#         # url_for("features.list", feature_name=feature_name, src_id=src_book,des_id=des_book),x)
#         #


# def feature_view_url(df, book):
#     columns=df.columns
#     for col in columns:
#         df[col]=df[col].astype(str)
#     for index,row  in df.iterrows():
#         for col in columns.difference(['avg']):
#             row[col]=generate_url(col,book,index,row[col])
#             df.set_value(index,col,row[col])
#             # print row[col]
#     # print df['unigram-bigram'].values
#     return df

current_context = {}
class ListView(MethodView):
    recent_values = {}

    def get_context(self, id):
        book = Book.objects.get_or_404(book_id=id)
        form = FeatureForm(request.form ,csrf_enabled=False)
        lang_tags=[(str(i+1),tag) for i,(tag,freq)in enumerate(dictionary.sort_dic_by_value(book.get_experiential_languages(),reverse=True))]
        form.exp_lang_tags.choices = lang_tags
        form_comment = model_form(RecommendationComment, exclude=(
            'comment_id', 'created_at', 'features', 'similar_books', 'dissimilar_books','exp_lang_tags'))(request.form)

        context = {
            "book": book,
            "form": form,
            "lang_tags":lang_tags,
            "selected_tags":{},
            "form_comment": form_comment,
            'most_similar': pd.DataFrame(),
            'most_similar_2': "",
            'least_similar_2': "",
            'least_similar': pd.DataFrame(),
            'features': [],
            'myfeatures':[],
            "comment_count": get_comments_count()
        }
        return context

    def get(self, id, slug):
        global current_context
        context = self.get_context(id)
        book=Book.objects.get_or_404(book_id=id)
        app = current_app._get_current_object()
        features = app.config['FEATURES']
        myfeatures = []
        for f in features:

            myfeatures.append(f)
        for f in myfeatures:
            # print ("##################")
            # print (f)

        #new api
        recommendation=Recommendation()
        feature_layer=FeaturesSimilarity(book,features)
        recommendation.add_layer(feature_layer)

        # most_similar, least_similar = get_n_similar_books(id, 10, features)

        ##most_similar, least_similar = recommendation.get_n_similar_books(10,kernel='cosine')
        # if format=='json':
        #    return jsonify(similar= most_similar.index.tolist() ,dissimilar=least_similar.index.tolist(),features= ['ALL'])

        ##most_similar['Title']= most_similar.index.map(build_url)
        ##most_similar['image_title'] = most_similar.index.map(build_img)
        ##most_similar['slug'] = most_similar.index.map(my_book_slug)

        ##least_similar['Title'] = least_similar.index.map(build_url)
        ##least_similar['image_title'] = least_similar.index.map(build_img)
       ## least_similar['slug'] = least_similar.index.map(my_book_slug)
 	n_most_similar = book.most_similar
        n_least_similar = book.least_similar
        most_similar=pickle.loads(codecs.decode(n_most_similar.encode(), "base64"))
        least_similar=pickle.loads(codecs.decode(n_least_similar.encode(), "base64"))
        context['most_similar'] = most_similar
        context['most_similar_2'] = n_most_similar
        context['least_similar_2'] = n_least_similar
        context['least_similar'] = least_similar
        #context['most_similar'] = most_similar
        #context['most_similar_2'] = codecs.encode(pickle.dumps(most_similar),"base64").decode()
        #context['least_similar_2'] = codecs.encode(pickle.dumps(least_similar),"base64").decode()
        #context['least_similar'] = least_similar
        context['features'] = myfeatures
        context['features'] = ['ALL']
        self.recent_values['most_similar'] = most_similar
        self.recent_values['least_similar'] = least_similar
        self.recent_values['features'] = context['features']
        self.recent_values['myfeatures'] = context['myfeatures']
        current_context = context

        return render_template('recommendations/list.html', **context)

    def post(self, id, slug=None):

        if slug:
            context = self.get_context(id)
            book=Book.objects.get_or_404(book_id=id)
            form = context.get('form')

      	    if form.validate():
                features_lst_1 = []
                myfeatures=[]
                number_of_reco=form.reco.data
                for field in form:
                    if field.name not in ['reco','exp_lang_tags']:
                        if field.data == True:
                            features_lst_1.append(field.name.replace('__', '-'))
                            myfeatures.append(field.name.replace('__', ' '))
                # print("Langauage TAga", form.exp_lang_tags.data)
                lang_tags=dict(context.get('lang_tags'))
                selected_tags=[ lang_tags.get(x) for x in form.exp_lang_tags.data ]

                features_lst = []
                char_ngram=['unigram','bigram','unigram__bigram','char_tri','char_4_gram','char_5_gram','char_tri__char_4_gram__char_5_gram']
                typed_char_ngram=['categorical_char_ngram_beg_punct','categorical_char_ngram_mid_punct','categorical_char_ngram_end_punct','categorical_char_ngram_multi_word','categorical_char_ngram_whole_word','categorical_char_ngram_mid_word','categorical_char_ngram_space_prefix','categorical_char_ngram_space_suffix','categorical_char_ngram_prefix','categorical_char_ngram_suffix']
                syntactic_features=['pos','phrasal','clausal','phr_cls','lexicalized','unlexicalized','gp_lexicalized','gp_unlexicalized']
                for i in features_lst_1:
                    if i=='char_ngram':
                        for j in char_ngram:
                            features_lst.append(j.replace('__', '-'))
                    if i=='syntactic_features':
                        for j in syntactic_features:
                            features_lst.append(j.replace('__', '-'))
                    if i=='typed_char_ngrams':
                        for k in typed_char_ngram:
                            features_lst.append(k.replace('__', '-'))
                    if i=='writing_density':
                        features_lst.append(i)
                    if i == 'readability':
                        features_lst.append(i)


                #new api
                recommendation=Recommendation()
                feature_layer=FeaturesSimilarity(book,features_lst)
                recommendation.add_layer(feature_layer)
                if selected_tags:
                    exp_lang_layer=ExperientialLanguageSimilarity(book,selected_tags)
                    recommendation.add_layer(exp_lang_layer)


                # most_similar, least_similar = get_n_similar_books(id, number_of_reco, features_lst)
                most_similar, least_similar =  recommendation.get_n_similar_books(number_of_reco,kernel='cosine')

                # if format=='json':
                #     return jsonify(similar= most_similar.index.tolist() ,dissimilar=least_similar.index.tolist(),features= features_lst)

                most_similar['Title'] = most_similar.index.map(build_url)
                most_similar['image_title'] = most_similar.index.map(build_img)
                most_similar['slug'] = most_similar.index.map(my_book_slug)

                least_similar['Title'] = least_similar.index.map(build_url)
                least_similar['image_title'] = least_similar.index.map(build_img)
                least_similar['slug'] = least_similar.index.map(my_book_slug)

                context['selected_tags']=selected_tags
                context['most_similar'] = most_similar
                context['least_similar'] = least_similar
                context['features'] = features_lst
                context['myfeatures'] = myfeatures
                self.recent_values['most_similar'] = most_similar
                self.recent_values['least_similar'] = least_similar
                self.recent_values['features'] = features_lst
                self.recent_values['myfeatures'] = myfeatures
                self.recent_values['selected_tags'] = selected_tags



                return render_template('recommendations/list.html', **context)




            return render_template('recommendations/list.html', **context)
        else:

            context = self.get_context(id)
            form_comment = context.get('form_comment')

            if form_comment.validate():
                print("Here in the comment")
                comment = RecommendationComment()
                form_comment.populate_obj(comment)
                book = context.get('book')
                comment.comment_id = len(book.comments)
                comment.similar_books = self.recent_values.get('most_similar')['Title'].tolist()
                comment.dissimilar_books = self.recent_values.get('least_similar')['Title'].tolist()
                comment.features = self.recent_values.get('features')
                comment.exp_lang_tags=self.recent_values.get('selected_tags')
                book.comments.append(comment)
                book.save()
                flash('Comment Saved.')
                # return "Comment Saved."
                return redirect(url_for('recommendations.list', id=id, slug=book.slug,format=format))
            flash('Could not save Comment.')



            return render_template('recommendations/list.html', **context)

class MyForm(Form):
    fi=HiddenField("F1")


@recommendations.route('/books/details/<id>/<index>/<myType>', methods=['POST','GET'])
def listDetail(id,index,myType):
    #global current_context
    if request.method == 'POST':
        m= request.form['most_similar_2']
        l=request.form['least_similar_2']
        most_similar = pickle.loads(codecs.decode(m.encode(),"base64"))
        least_similar = pickle.loads(codecs.decode(l.encode(), "base64"))

        de=DetailView()
        result1 = de.drawChart(id)
        pi_plot1 = result1[0]

        result2 = de.drawChart(index)
        pi_plot2 = result2[0]

        if myType=="0":
            fig = myChart(most_similar,index)
        else:
            fig = myChart(least_similar,index)

        bar_plot = plot([fig], output_type='div')
        book = Book.objects.get_or_404(book_id=id)
        book_des = Book.objects.get_or_404(book_id=index)

        print myType
        if myType=="0":
            avg=getAvg(most_similar,index)
        else:
            avg=getAvg(least_similar,index)
        x=[]
        for col in most_similar.columns:
            if col != "Title" and col != "avg" and col != "explan_avg" and col != "feature_avg" and col!="image_title" and col!="slug":
                x.append(col.replace('_',' '))

    return render_template('recommendations/details.html',pi_plot1=Markup(pi_plot1) ,pi_plot2=Markup(pi_plot2),my_plot=Markup(bar_plot), book=book, book_des=book_des, avg=avg, features=x)



@recommendations.route('/recommendations/<id>/kernels', methods=['GET', 'POST'])
def all_kernels_recommendations(id):
    n=20
    KERNELS = ['cosine','linear','polynomial','rbf','laplacian','sigmoid']

    result=defaultdict(lambda: defaultdict(list))

    book = Book.objects.get_or_404(book_id=id)
    app = current_app._get_current_object()
    features = app.config['FEATURES']
    myfeatures=[]
    for f in features:
        myfeatures.append(f)
    # for f in myfeatures:
        # print ("##################")
        # print f
    recommendation=Recommendation()
    feature_layer=FeaturesSimilarity(book,features)
    recommendation.add_layer(feature_layer)
    for kernel in KERNELS:
        most_similar, least_similar = recommendation.get_n_similar_books(n,kernel=kernel)
        result[kernel]['most_similar']=most_similar.index.map(build_url)
        result[kernel]['least_similar']=least_similar.index.map(build_url)

   


    return render_template('recommendations/kernels.html', book=book,recos=result,n_reco=n)

def findSimilarity0():
    print ("inja")
    f = open("/home/ritual/Desktop/resultSimilarity_new.txt", "w")
    books = Book.objects(is_active=True).timeout(False).all()
    for book in books:
        app = current_app._get_current_object()
        features = app.config['FEATURES']

        recommendation=Recommendation()
        feature_layer=FeaturesSimilarity(book,features)
        recommendation.add_layer(feature_layer)

        most_similar, least_similar = recommendation.get_n_similar_books(10,kernel='cosine')

        z=[]
        t=[]
        for index, row in most_similar.iterrows():
            y = []
            for col in most_similar.columns:
                if  col != "Title" and col != "avg" and col != "explan_avg" and col != "feature_avg" and col != "image_title" and col != "slug":
                    y.append(row[col])

            z.append(y)
        print >> f, "most_similar: "+ str(z)
        for index, row in least_similar.iterrows():
            x = []
            for col in least_similar.columns:
                if col != "Title" and col != "avg" and col != "explan_avg" and col != "feature_avg" and col != "image_title" and col != "slug":
                    x.append(row[col])
            t.append(x)
        print >> f, "least_similar: " + str(t)

        time.sleep(3)

def findSimilarity0():
    print ("inja")
    counter=0 
    books = Book.objects(is_active=True).timeout(False).all()
    for book in books:
	print (counter)
	counter=counter+1
	print( "*****")
	if book.most_similar is None or book.least_similar is None:
        	app = current_app._get_current_object()
        	features = app.config['FEATURES']

        	recommendation=Recommendation()
        	feature_layer=FeaturesSimilarity(book,features)
        	recommendation.add_layer(feature_layer)

        	most_similar, least_similar = recommendation.get_n_similar_books(10,kernel='cosine')
        #print most_similar
       		most_similar['Title']= most_similar.index.map(build_url)
        	most_similar['image_title'] = most_similar.index.map(build_img)
        	most_similar['slug'] = most_similar.index.map(my_book_slug)

        	least_similar['Title'] = least_similar.index.map(build_url)
        	least_similar['image_title'] = least_similar.index.map(build_img)
        	least_similar['slug'] = least_similar.index.map(my_book_slug)
        	new_most_similar=codecs.encode(pickle.dumps(most_similar),"base64").decode()
        	new_least_similar=codecs.encode(pickle.dumps(least_similar),"base64").decode()
        	Book.objects(book_id=book.book_id).update_one(most_similar=new_most_similar)
        	Book.objects(book_id=book.book_id).update_one(least_similar=new_least_similar)

		time.sleep(3)


reco_view = ListView.as_view('list')
reco_comment = ListView.as_view('comment')

recommendations.add_url_rule('/recommendations/<id>/<slug>/',view_func=reco_view, methods=['GET', 'POST'])
recommendations.add_url_rule('/recommendations/<id>/', defaults={'slug': None,'format':None}, view_func=reco_comment,
                             methods=['GET', 'POST'])

	

