# -*- coding: utf-8 -*-
import random

from bookweb.models import *
from os import listdir
from os.path import isfile, join
import os
import codecs
import json
import datetime
import csv
import re
from pymongo import MongoClient
import StringIO
from datetime import datetime

__author__ = 'suraj'


WIERED_UNICODE = re.compile('[ziч黜溫蠶蠶醬欖驚欖蠶器ಕ್ಹ黔醬]', re.UNICODE)


def chinese_characters(text):
    for word in text:
        if (ord(u'\u4e00') <= ord(word[0]) <= ord(u'\u9fff')):
            raise ValueError('Chinese characters in the text')
    return True


def private_user_area(text):
    for word in text:
        if (ord(u'\ue000') <= ord(word[0]) <= ord(u'\uf8ff')):
            raise ValueError('Private characters in the text')
    return True


def get_comments_count():
    comments_count = Book.objects().aggregate({"$project": {"comments": 1}},
                                                            {"$unwind": "$comments"}, {
                                                                "$group": {"_id": "result",

                                                                           "count": {"$sum": 1}}})

    c = 0
    for row in comments_count:
        c = row['count']
        break
    return c


def read_book(fpath, encoding='latin1'):
    content = ''
    with codecs.open(fpath, 'r', encoding=encoding) as f_in:
        content = f_in.read()
    return content


def google_book_mapping(row):
    data={}
    data['book_id'] = row['id']

    data['title'] = row['volumeInfo']['title'] + ', ' + row['volumeInfo']['subtitle'] if row[
        'volumeInfo'].has_key('subtitle')  else row['volumeInfo']['title']
    data['slug'] = "-".join(data['title'].replace(',', '').split())
    data['tags'] = []
    data['authors'] = [author for author in row['volumeInfo']['authors']] if row[
        'volumeInfo'].has_key('authors') else []

    data['published_date'] = row['volumeInfo']['publishedDate'] if row['volumeInfo'].has_key(
        'publishedDate') else None
    data['publisher'] = row['volumeInfo']['publisher']
    data['url'] = row['accessInfo']['webReaderLink']
    data['isbn_10'] = row['volumeInfo']['industryIdentifiers'][0]['identifier'] if \
        row['volumeInfo']['industryIdentifiers'][0]['type'] == 'ISBN_10' else \
        row['volumeInfo']['industryIdentifiers'][1]['identifier']
    data['isbn_13'] = row['volumeInfo']['industryIdentifiers'][1]['identifier'] if \
        row['volumeInfo']['industryIdentifiers'][1]['type'] == 'ISBN_13' else \
        row['volumeInfo']['industryIdentifiers'][0]['identifier']

    data['description'] = row['volumeInfo']['description'] if row['volumeInfo'].has_key(
        'description') else " "
    data['page_count'] = int(row['volumeInfo']['pageCount'])
    data['cover_image'] = 'http://books.google.ru/books/content?id={id}&printsec=frontcover&img=1&zoom=1&source=gbs_api'.format(
        id=data['book_id'])
    data['self_link'] = row['selfLink']

    category = set()
    if row['volumeInfo'].has_key('categories'):
        for d in row['volumeInfo']['categories']:
            for genre in d.split('/'):
                category.add(genre.strip())
        data['genre'] = list(category)
    else:
        data['genre'] = []

    return data


def insert_avg_rate ():
    datafile = open('./train_test_split_goodreads_avg_rating.yaml', 'r')
    arr = {}
    for line in datafile:
        data = line.split("_")
        bookId = data[0].replace('- ', '')
        avg = line.split()[-1]
        avgRate=float(avg)
        arr[bookId] = avgRate
    return arr

def save_new_google_book():
    idArr = {}

    path = './bad-books.txt'
    pathToContent = './bookProject/OCR/final-book/'
    f = open(path)
    content=f.readlines()
    genrelist = []
    authorslist = []
    title = ""
    publish_date = ""
    publisher = ""
    genre = ""
    authors = ""
    Description = ""
    book_id = ""
    ISBN13=""
    ISBN10=""
    page_count=""
    avg_rating=""
    contentBook=""

    k=0
    for line in content:

        if (line.strip() == "============"):
            if genre != None:
                tempGenre = genre.split(",")
                numOfGenre = len(tempGenre)
                for i in range(numOfGenre):
                    genrelist.append(tempGenre[i])
            if authors != None:
                tempAuthors = authors.split(",")
                numOfAuthors = len(tempAuthors)
                for i in range(numOfAuthors):
                    authorslist.append(tempAuthors[i])
            #print os.path.isfile(pathToContent+book_id+"_"+book_id+".txt")
            if (os.path.isfile(pathToContent+book_id+"_"+book_id+".txt")):
                f2 = open(pathToContent+book_id+"_"+book_id+".txt")
                contentBook = f2.read()


            book = GutenbergBook(book_id=book_id, title=title,
                                 slug="-".join(title.replace('(', '').replace(')', '').split()),
                                 publisher=publisher, authors=authorslist, genre=genrelist, description=Description,
                                 isbn_10=ISBN10,isbn_13=ISBN13, avg_rating= avg_rating, page_count= page_count,publish_date=publish_date,
                                 content=contentBook,book_condition="failure")


            if len(Book.objects)>0:
                if Book.objects.filter(book_id=book_id).first() is None:
                    if (contentBook != ""):
                        print(title)
                        book.save()
            else:
                book.save()
            genrelist = []
            authorslist = []
            title = ""
            publish_date = ""
            publisher = ""
            genre = ""
            authors = ""
            Description = ""
            book_id = ""
            ISBN13 = ""
            ISBN10 = ""
            page_count = ""
            avg_rating = ""
            contentBook = ""

        else:
            data = line.strip().split(":")
            if (data[0] == "book_id"):
                book_id = data[1]
            if (data[0] == "title"):
                title = data[1]
            if (data[0] == "ISBN0"):
                if len(str(data[1])) == 13:
                    ISBN13 = data[1]
                if len(str(data[1])) == 10:
                    ISBN10 = data[1]
            if (data[0] == "ISBN1"):
                if len(str(data[1])) == 13:
                    ISBN13 = data[1]
                if len(str(data[1])) == 10:
                    ISBN10 = data[1]
           # if (data[0] == "publish-date"):
            #    publish_date=datetime.strptime(data[1], ' %Y-%m-%d')
            if (data[0] == "publisher"):
                publisher = data[1]
            if (data[0] == "page_count"):
                if (data[1] != " null"):
                    page_count = int(data[1])
                else:
                    page_count = None 
            if (data[0] == "genre"):
                genre = data[1]
            if (data[0] == "avg_rating"):
                if(data[1] != " null"):
                    avg_rating = float(data[1])
                else:
                    avg_rating = None
            if (data[0] == "ratingCount"):
                ratingCount = data[1]
            if (data[0] == "Author(s)"):
                authors = data[1]
            if (data[0] == "Description"):
                Description = data[1]





def save_gutenberg_books():

    idArr = {}
    avgArr = insert_avg_rate()
    path = './book_eacl/'
    allSubdirs = [d for d in os.listdir(path) if os.path.isdir(path + d)]
    for y in allSubdirs:
        mypath = path + y + "/success/"
        mypath2 = path + y + "/failure/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        for x in onlyfiles:
            data = x.split("_")
            bookId = data[0]
            bookName = data[1].replace('+', ' ')
            bookName = bookName.replace('.txt', '')
            f = open(mypath + x)
            content = f.read()
            rate=avgArr[bookId]

            book = GutenbergBook(book_id=bookId, title=bookName,
                                 slug="-".join(bookName.replace('(', '').replace(')', '').split()), content=content,
                                 genre=[y], book_condition="success", avg_rating=rate)

            if idArr.get(bookId) is None:
                book.save()
            else:
                Ngenre= Book.objects.get(book_id=bookId)['genre']
                print("++++++++++++++ Ngenre:")
                print(Ngenre)
                Ngenre.append(y)
                print("------------------NNgenra:")
                print(Ngenre)
                Book.objects(book_id=bookId).update_one(set__genre=Ngenre)
            idArr[bookId]=1

        onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
        for x in onlyfiles2:
            data = x.split("_")
            bookId = data[0]
            bookName = data[1].replace('+', ' ')
            bookName = bookName.replace('.txt', '')
            f = open(mypath2 + x)
            content = f.read()
            rate = avgArr[bookId]
            book = GutenbergBook(book_id=bookId, title=bookName,
                                 slug="-".join(bookName.replace('(', '').replace(')', '').split()), content=content,
                                 genre=[y], book_condition="failure", avg_rating=rate)

            # if idArr.get(bookId) is None:
            if idArr.get(bookId) is None:
                book.save()
            else:
                Ngenre= Book.objects.get(book_id=bookId)['genre']
                print("++++++++++++++N:")
                print(Ngenre)
                Ngenre.append(y)
                print("------------------NN:")
                print(Ngenre)
                Book.objects(book_id=bookId).update_one(set__genre=Ngenre)

            idArr[bookId] = 1

def delteComments():
    books = Book.objects(is_active=True).timeout(False).all()
    for book in books:
        Book.objects(book_id=book.book_id).update_one(comments=[])


def getInfoToFile():


    import pandas as pd
    data=[]

    for book in Book.objects(is_active=True):
        data.append({'ID': book.book_id,'Title':book.title.strip('\r\n'),
                     'Avg_Rating':book.avg_rating,
                     'Authors':": ".join(author.strip('\r\n') for author in  book.authors),
                     'Genre':": ".join(b.strip('\r\n') for b in book.genre)})

    df=pd.DataFrame(data)


    df.to_csv('eacl_data.csv', index=False, encoding='utf-8')
