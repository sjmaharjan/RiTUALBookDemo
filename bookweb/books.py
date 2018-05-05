__author__ = 'suraj'

import cgi

import array
from markupsafe import Markup

from flask import Blueprint, request, redirect, render_template, url_for, flash
from flask.views import MethodView
from flask_mongoengine.wtf.orm import model_form

from bookweb.models import Book, GutenbergBook, GoogleBook, Comment,BookStatus
from flask import current_app
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired
from bookweb.utils import  get_comments_count
from plotly.offline import plot
import matplotlib.pyplot as plt
from plotly.graph_objs import Scatter
import plotly.plotly as py
import plotly.graph_objs as go
books = Blueprint('books', __name__, template_folder='templates')


class ListView(MethodView):
   def get(self, page=1):
      app = current_app._get_current_object()
      books = Book.objects(is_active=True).paginate(page=page, per_page=app.config['BOOKS_PER_PAGE'])
      book_source = Book.objects(is_active=True).aggregate({"$group": {"_id": "$_cls", "count": {"$sum": 1}}})
      book_source_genre= Book.objects(is_active=True).aggregate({"$unwind":"$genre"},{"$group": {"_id":"$genre", "count": {"$sum": 1}}})

      return render_template('books/list.html', books=books, book_src=book_source, book_src_g=list(book_source_genre) ,comment_count=get_comments_count())

class BookSourceListView(MethodView):
    def get(self, name, page=1):
        app = current_app._get_current_object()
        books = Book.objects(_cls=name, is_active=True).paginate(page=page, per_page=app.config['BOOKS_PER_PAGE'])
        book_source = Book.objects(is_active=True).aggregate({"$group": {"_id": "$_cls", "count": {"$sum": 1}}})
        book_source2 = Book.objects(is_active=True).aggregate({"$unwind":"$genre"},{"$group": {"_id": "$genre", "count": {"$sum": 1}}})
        return render_template('books/list.html', books=books, book_src=list(book_source), book_src_g=list(book_source2),comment_count=get_comments_count())

class GenreListView(MethodView):
    def get(self, name, page=1):
        app=current_app._get_current_object()
        book_source2 = Book.objects(is_active=True).aggregate({"$group": {"_id": "$_cls", "count": {"$sum": 1}}})
        books = Book.objects(genre=name, is_active=True).paginate(page=page, per_page=app.config['BOOKS_PER_PAGE'])
        book_source= Book.objects(is_active=True).aggregate({"$unwind":"$genre"},{"$group": {"_id": "$genre", "count": {"$sum": 1}}})
        return render_template('books/list.html', books=books, book_src=book_source2,book_src_g=book_source,comment_count=get_comments_count())

class DetailView(MethodView):
    form = model_form(Comment, exclude=['comment_id', 'created_at'])

    def get_context(self, id, slug):
        book = Book.objects.get_or_404(book_id=id)
        book_source = Book.objects(is_active=True).aggregate({"$group": {"_id": "$_cls", "count": {"$sum": 1}}})
        form = self.form(request.form)


        context = {
            "book": book,
            "book_src": book_source,
            "form": form,

            "comment_count":get_comments_count()
        }
        return context


    def drawChart(self, id):
        book = Book.objects.get_or_404(book_id=id)
        myContent=book.content
        # book = Book.objects.get_or_404(book_id=id)
        dictOfEmotion=sentimentAnalyse(myContent)
        x=[1, 2, 3,4,5,6,7,8,9,10]
        joy = []
        anger = []
        anticipation = []
        disgust = []
        fear = []
        negative = []
        positive = []
        sadness = []
        surprise = []
        trust = []
        forPi = 0
        joyForPi = []
        angerForPi = []
        anticipationForPi = []
        disgustForPi = []
        fearForPi = []
        negativeForPi = []
        positiveForPi = []
        sadnessForPi = []
        surpriseForPi = []
        trustForPi = []
        valueForPi = []

        changeStringToInputForPi = {'joy': joyForPi, 'anger': angerForPi, 'anticipation': anticipationForPi,
                                    'disgust': disgustForPi, 'fear': fearForPi,
                                    'negative': negativeForPi, 'positive': positiveForPi, 'sadness': sadnessForPi,
                                    'surprise': surpriseForPi,
                                    'trust': trustForPi}

        listOfEmotionForPi = ['joy', 'anger', 'anticipation', 'disgust', 'fear', 'sadness', 'surprise', 'trust']
        changeStringToInput = {'joy': joy, 'anger': anger, 'anticipation': anticipation, 'disgust': disgust,
                               'fear': fear,
                               'negative': negative, 'positive': positive, 'sadness': sadness, 'surprise': surprise,
                               'trust': trust}
        listOfEmotion = ['joy', 'anger', 'anticipation', 'disgust', 'fear', 'negative', 'positive', 'sadness',
                         'surprise', 'trust']
        #trend
        for x in listOfEmotion:
            for iterator in range(10):
                changeStringToInput.get(x).append(dictOfEmotion.get(x).get(iterator))
        #pi

        for item in listOfEmotionForPi:
            for element in changeStringToInput.get(item):
                if element is not None:
                    forPi=element+forPi
            changeStringToInputForPi.get(item).append(forPi)
            forPi=0

        for item in listOfEmotionForPi:
            valueForPi.append(changeStringToInputForPi.get(item)[0])
        #diagram
        trace = go.Pie(labels=listOfEmotionForPi, values=valueForPi)
    	print "##########"
        print valueForPi
        print "##########"
        pi_plot=plot([trace],output_type='div')

        my_plot_div=plot([Scatter(x=x,y=joy,name='joy'),Scatter(x=x,y=fear,name='fear'),Scatter(x=x,y=anger,name='anger'),Scatter(x=x,y=anticipation,name='anticipation'),Scatter(x=x,y=disgust,name='disgust'),
                          Scatter(x=x,y=negative,name='negative'),Scatter(x=x,y=positive,name='positive'),Scatter(x=x,y=sadness,name='sadness'),Scatter(x=x,y=surprise,name='surprise'),Scatter(x=x,y=trust,name='trust')]
                         ,output_type='div')
        return [pi_plot,my_plot_div]


    def get(self, id, slug):
        context = self.get_context(id, slug)
        result=self.drawChart(id)
        pi_plot=result[0]
        my_plot_div=result[1]

        return render_template('books/detail.html',p=Markup(my_plot_div),pichart=Markup(pi_plot), **context)

    def post(self, id, slug):
        context = self.get_context(id, slug)
        form = context.get('form')
        if form.validate_on_submit():
            if form.validate():
                comment = Comment()
                form.populate_obj(comment)

                book = context.get('book')
                comment.comment_id = len(book.comments)
                book.comments.append(comment)
                book.save()
                flash('Comment Saved.')
                # return "Comment Saved."

                # socketio.emit('msg', {'count':get_comments_count()},namespace='/booxby', broadcast=True)
                return redirect(url_for('books.detail', id=id, slug=slug))
            flash('Could not save Comment.')
            return render_template('books/detail.html', **context)

def makeDict():
    d = {}
    d2 ={}
    with open("/home/mahsa/ml-master2/ml-master/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt") as f:
        first_line= f.readline()
        preWord=first_line.split()[0].strip()
        for line in f:
            if line.split()[0].strip()==preWord:
                key=line.split()[1].strip()
                val=line.split()[2]
                d[key] = val
            else:
                key2=preWord
                d2[key2] = d
                d={}
                key = line.split()[1].strip()
                val = line.split()[2]
                d[key] = val
                preWord=line.split()[0].strip()
        key2 = preWord
        d2[key2] = d
    return d2

def sentimentAnalyse(myContent):
    d=makeDict()
    joy = {}
    anger= {}
    anticipation ={}
    disgust ={}
    fear ={}
    negative={}
    positive={}
    sadness={}
    surprise={}
    trust={}
    dictOfEmotion={'joy':joy,'anger':anger,'anticipation':anticipation,'disgust':disgust,'fear':fear,'negative':negative,'positive':positive,'sadness':sadness,'surprise':surprise,'trust':trust}
    listOfEmotion=['joy','anger','anticipation','disgust','fear','negative','positive','sadness','surprise','trust']
    temp=0
    index=0
    #with open("/home/ritual/Desktop/ml-master/book_eacl/Fiction/success/144_the+voyage+out.txt") as f:
    #    content=f.read()
    content=myContent
    num_lines = sum(1 for line in content.splitlines())
    total=float(num_lines)/10
    eachIter=int(total)
    remain=total-int(total)
    firstIteration=eachIter+remain*10
    for line in content.splitlines():
        line = line.strip()
        if temp <eachIter:
            temp = temp + 1
        elif temp == eachIter:
            temp=0
            index=index+1
        wordInLine=line.split()
        for word in wordInLine:
            d2=d.get(word)
            if d2 is not None:
                for j in listOfEmotion:
                    val2=d2.get(j)
                    if val2 is not None:
                        prevCount=dictOfEmotion.get(j).get(index)
                        if prevCount is not None:
                            currentCount=int(prevCount)+int(val2)
                        else:
                            currentCount=int(val2)
                        dictOfEmotion.get(j)[index]=currentCount
    for k in listOfEmotion:
        print(k)
        print("------------------>")
        print dictOfEmotion.get(k)
    return dictOfEmotion

class SearchForm(Form):
    query = StringField('Search', validators=[DataRequired()])


class SearchView(MethodView):
    def get(self, query=''):
        books = Book.objects(title__icontains=query)
        book_source = Book.objects(is_active=True).aggregate({"$group": {"_id": "$_cls", "count": {"$sum": 1}}})
        return render_template('books/search.html', books=books, query=query, book_src=list(book_source))

    def post(self):
        form = SearchForm(request.form, csrf_enabled=False)
        if request.method == 'POST':  # and form.validate():
            query = form.query.data
            return redirect(url_for('books.search', query=query))
        return redirect(url_for('books.list'))


class GenerateView(MethodView):
    def get(self, slug):
        book = Book.objects.get_or_404(slug=slug)
        return render_template('books/detail.html', book=book)





#
@books.route('/<id>/<slug>/updateContent', methods=['POST'])
def updateContent(id, slug):
    book = Book.objects.get_or_404(book_id=id)
    book.update
    Book.objects(book_id=id).update_one(content=request.form['text'])
    return redirect(url_for('books.detail', id=id, slug=slug))

@books.route('/<id>/<slug>/activate', methods=['GET', 'POST'])
def activate(id, slug):
    book = Book.objects.get_or_404(book_id=id)
    book.update(set__status=BookStatus.ACTIVE)
    book.update(set__is_active=True)
    flash('Book is now active.')

    return redirect(url_for('books.detail', id=id, slug=slug))


@books.route('/<id>/<slug>/deactivate', methods=['GET', 'POST'])
def deactivate(id, slug):

    book = Book.objects.get_or_404(book_id=id)
    book.update(set__is_active=False)
    book.update(set__status=BookStatus.INACTIVE)
    flash('Book removed.')

    return redirect(url_for('books.detail', id=id, slug=slug))


@books.route('/books/removed', methods=['GET'])
def removed():
    books = Book.objects(is_active=False)
    return render_template('books/remove.html', books=books,comment_count=get_comments_count())

@books.route('/books/about', methods=['GET'])
def about():

    return render_template('books/about.html')






# Register the urls
book_view = ListView.as_view('list')
book_search = SearchView.as_view('search')
book_source = BookSourceListView.as_view('source')
book_source_genre = GenreListView.as_view('test')

books.add_url_rule('/', view_func=book_view, methods=['GET', ], defaults={'page': 1})
books.add_url_rule('/page/<int:page>', view_func=book_view)
books.add_url_rule('/<id>/<slug>/', view_func=DetailView.as_view('detail'))

books.add_url_rule('/search', view_func=book_search, methods=['POST'])
books.add_url_rule('/search/<query>', view_func=book_search, methods=['GET', 'POST'])
books.add_url_rule('/generate', view_func=GenerateView.as_view('generate'))
books.add_url_rule('/source/<name>/', view_func=book_source, defaults={'page': 1}, methods=['GET', ])
books.add_url_rule('/source/<name>/page/<int:page>', view_func=book_source, methods=['GET', ])
books.add_url_rule('/test/<name>/', view_func=book_source_genre, defaults={'page': 1}, methods=['GET', ])
books.add_url_rule('/test/<name>/page/<int:page>', view_func=book_source_genre, methods=['GET', ])

