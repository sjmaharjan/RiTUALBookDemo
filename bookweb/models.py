# -*- coding: utf-8 -*-
import datetime
from collections import defaultdict

from flask import url_for
from bookweb import db
import nltk

__author__ = 'suraj'


class BookStatus:
    INACTIVE = 0
    UNPROCESSED = 1
    ACTIVE = 2


class Comment(db.EmbeddedDocument):
    comment_id = db.IntField(verbose_name='Comment ID',required=True,help_text='Comment ID')
    created_at = db.DateTimeField(default=datetime.datetime.utcnow, required=True,help_text='Comment Created at',verbose_name='Comment Create Date')
    body = db.StringField(verbose_name="Comment", required=True,help_text='Comment')
    author = db.StringField(verbose_name="Name", max_length=255, required=True,help_text='Commenter')

    @property
    def comment_type(self):
        return self.__class__.__name__

    meta = {
        'allow_inheritance': True,
        'ordering': ['created_at']

    }


class RecommendationComment(Comment):
    similar_books = db.ListField(db.StringField())
    dissimilar_books = db.ListField(db.StringField())
    features = db.ListField(db.StringField())
    exp_lang_tags=db.ListField(db.StringField())


class Book(db.DynamicDocument):
    """
    Book object relation mapping with mongo collection book
    """

    book_id = db.StringField(verbose_name='Book ID',max_length=255, required=False,help_text='Google Book ID or Gutenberg Book Id or Empty')
    title = db.StringField(verbose_name="Title",max_length=255, required=True,help_text='Book Title')
    slug = db.StringField(verbose_name="Slug",max_length=255, required=True,help_text='Slug Title')
    tags = db.ListField(db.StringField(max_length=30,verbose_name='Tag',help_text='Tags'),help_text='Tags ',verbose_name='Tags ')
    synthetic_tags_5 = db.ListField(db.StringField(max_length=30, verbose_name='S_Tag_5', help_text='S_Tag_5'),
                                    help_text='S_Tag_5 ', verbose_name='Synthetic Tags from 5 most similar books')
    synthetic_tags_10 = db.ListField(db.StringField(max_length=30, verbose_name='S_Tag_10', help_text='S_Tag_10'),
                                     help_text='S_Tag_10 ', verbose_name='Synthetic Tags from 10 most similar books')
    synthetic_tags_15 = db.ListField(db.StringField(max_length=30, verbose_name='S_Tag_15', help_text='S_Tag_15'),
                                     help_text='S_Tag_15 ', verbose_name='Synthetic Tags from 15 most similar books')
    authors = db.ListField(db.StringField(max_length=255,verbose_name='Authors',help_text='Authors'),help_text='Authors ',verbose_name='Authors ')
    genre = db.ListField(db.StringField(max_length=255,verbose_name='Genre',help_text='Genre'),help_text='Genre ',verbose_name='Genres ')
    published_date = db.DateTimeField(verbose_name="Published Date",help_text='Book Published Date')
    publisher = db.StringField(verbose_name="Publisher",max_length=255,help_text='Book Publisher')
    url = db.URLField(verbose_name="Book URL",help_text='Online Link to Book ')
    isbn_10 = db.StringField(verbose_name="ISBN 10",max_length=10,help_text='Book ISBN 10')
    isbn_13 = db.StringField(verbose_name="ISBN 13",max_length=13,help_text='Book ISBN 13')
    content = db.StringField(verbose_name="Content",help_text='Book Content')
    is_active = db.BooleanField(verbose_name="Active",default=False,help_text='Active')  # need to get rid of this field and use status
    comments = db.ListField(db.EmbeddedDocumentField('Comment',help_text='Comments ',verbose_name='Comments '),help_text='Comments ',verbose_name='Comments ')
    skip_pages = db.IntField(verbose_name="Skip Page",help_text='From which page to start extract content',default=0)
    book_file_name = db.StringField(verbose_name="Book File Name",help_text='Book uploaded to')
    status = db.IntField(verbose_name="Book Status",default=BookStatus.UNPROCESSED,help_text='Book Status')
    page_count = db.IntField(help_text='Total Pages ',verbose_name='Total Pages ')
    book_condition = db.StringField(verbose_name="condition",max_length=255, required=False,help_text='Book is success or failure')
    avg_rating = db.FloatField(help_text='Average Rating ', verbose_name='Average Rating')
    most_similar= db.StringField(verbose_name="Most Similar Books",help_text='Most Similar Books')
    least_similar= db.StringField(verbose_name="Least Similar Books",help_text='Least Similar Books')
    def get_absolute_url(self):
        return url_for('book', kwargs={"slug": self.slug})

    def __unicode__(self):
        return self.title

    @property
    def book_type(self):
        return self.__class__.__name__

    def get_n_sentences(self, content, n=1000):
        full_text = ""
#	content=content.decode('utf-8')
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            sentences = nltk.sent_tokenize(line)
            full_text += " ".join(sentences[:n - len(nltk.sent_tokenize(full_text))]) + "\n"
            if len(nltk.sent_tokenize(full_text)) >= n:
                break
        return full_text



    def populate_obj(self,**kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


    def get_experiential_languages(self):
        exp_lang=defaultdict(int)
        for tag in self.tags:
            try:
                phrase,freq=tag.rsplit('(',1)
                exp_lang[phrase.lower()]=int(freq.replace(')','').strip())
            except ValueError:

                print ("Split error for tag %s for book %s"%(tag, self.title))
        return exp_lang


    def get_synthetic_tags(self, n_books):
        if n_books == 5:
            return self.synthetic_tags_5
        elif n_books == 10:
            return self.synthetic_tags_10
        elif n_books == 15:
            return self.synthetic_tags_15
        

    def has_experiential_languages(self):
        return len([ x for x in self.tags if x.strip()])

    def get_experiential_languages_for_ui(self):
        exp_lang = defaultdict(int)
        exp_lang_filtered = defaultdict(int)
        phrase_without_freq_one = 0

        for tag in self.tags:
            try:
                phrase, freq = tag.rsplit('(', 1)
                freq = int(freq.replace(')', '').strip())

                if freq > 1:
                    phrase_without_freq_one += 1
                    exp_lang_filtered[phrase.lower()] = freq

                exp_lang[phrase.lower()] = freq

            except ValueError:
                print("Split error for tag %s for book %s" % (tag, self.title))

        if phrase_without_freq_one >= 3:
            return exp_lang_filtered
        else:
            return exp_lang



    meta = {
        'allow_inheritance': True,
        'indexes': ['id', 'slug'],

    }


class GoogleBook(Book):
    """Extends Book class and adds fields specific with google books

    """
    description = db.StringField(verbose_name="Description",help_text='Description ')

    cover_image = db.URLField(help_text='Cover Image URL ',verbose_name='Cover Image URL ')
    self_link = db.URLField(help_text='Google Book API URL ',verbose_name='Google Book API URL  ')





class GutenbergBook(Book):
    """Extends Book class and adds fields specific with gutenberg books

    """
    book_source = db.StringField(max_length=255, default="Practical Classics",help_text='Book From',verbose_name='Book From')

    def get_n_sentences_gutenberg(self, content, n=1000):
        full_text = ""
#	content=content.decode('latin1')
        for line in content.splitlines():
            line = line.strip()
            if not line:
                full_text += "\n\n"
                continue
            sentences = nltk.sent_tokenize(line)
            full_text += " ".join(sentences[:n - len(nltk.sent_tokenize(full_text))]) + " "
            if len(nltk.sent_tokenize(full_text)) >= n:
                break
        return full_text


class Authors(Book):
    """Extends Book class and adds fields specific with Book provided by authors

    """
    book_source = db.StringField(max_length=255, default="Authors",help_text='Book From ',verbose_name='Book From ')

    description = db.StringField(verbose_name="Description",help_text='Description')
    cover_image = db.URLField(help_text='Cover Image Link ',verbose_name='Cover Image Link')
    self_link = db.URLField(help_text='Google Book API URL ',verbose_name='Google Book API URL  ')
