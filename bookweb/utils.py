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


def save_googlebook_to_db(files, type='GoogleBook'):
    for file in os.listdir(files):
        if file.endswith('.json'):
            with codecs.open(os.path.join(files, file), 'r', encoding='utf-8') as f_in:
                row = json.load(f_in)
                data=google_book_mapping(row)
                if type == 'GoogleBook':
                    data_path = '/home/booxby/PycharmProjects/Skywriterx/data'
                    content = read_book(os.path.join(data_path, file.replace('.json', '.txt')), encoding='utf-8')
                    book = GoogleBook(content=content,**data)
                elif type == 'Authors':
                    content = read_book(os.path.join(files, file.replace('.json', '.txt')), encoding='utf-8')
                    book = Authors(content=content,**data)
                book.save()


#def save_gutenberg_books():
#    path = '/home/booxby/Downloads/practical.csv'

#    with open(path, 'r') as f_in:
#        csv_reader = csv.DictReader(f_in)
#        for row in csv_reader:
#            print row['Book_id']
#            content = read_book(row['Book_id'])

#            book = GutenbergBook(book_id=row['Book_id'].replace('.txt', ''), title=row['Name '],
#                                 slug="-".join(row['Name '].replace('(', '').replace(')', '').split()),
#                                 authors=[row['Author']], url=row['URL'],
#                                 content=content)
#            book.save()
def insert_avg_rate ():
    datafile = open('/home/mahsa/ml-master2/ml-master/book_eacl/train_test_split_goodreads_avg_rating.yaml', 'r')
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

    path = '/home/ritual/Desktop/bookProject/finalResult/bad-books.txt'
    pathToContent = '/home/ritual/Desktop/bookProject/OCR/final-book/'
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
    path = '/home/mahsa/ml-master2/ml-master/book_eacl/'
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

def saveGutenbergAuthor():
    datafile = open('/home/mahsa/ml-master2/ml-master/book_eacl/finalFile', 'r')
    for line in datafile:
        data=line.split(":")
        bookId=data[0]
        author=data[1]
        Book.objects(book_id=bookId).update_one(authors=[author])

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
     #f = open('csvfile.csv', 'w')
    # with codecs.open('/home/ritual/Desktop/ml-master/book_eacl/outFile.txt', 'wb',encoding='utf-8') as f:
    #
    #     for book in Book.objects(is_active=True):
    #
    #         f.write(book.book_id)
    #         f.write(":")
    #         f.write(book.title)
    #         f.write(":")
    #         for i in book.genre:
    #             f.write(''.join(i))
    #             f.write(",")
    #         f.write(":")
    #         f.write(str(book.avg_rating))
    #         f.write(":")
    #         for j in book.authors:
    #             f.write(''.join(j).rstrip())
    #             f.write(",")
    #         f.write(":")
    #     f.write('\n')
    # f.close()

def save_authors_book():
    mapping = {  # 'End Game': ['Josh Conviser', 'isbnnotknown', '', '', '', '559','',
        #              '/home/suraj/resouces/book_holly/EndGame.Conviser.10.29.15_1000.txt'],
        # 'THE SOUND OF BLUE': ['Holly Lynn Payne', 'yiIrAQAAIAAJ', '0982279752', '9780982279755',
        #                       'Skywriter Books; revised edition (June 1, 2015)', '258', '4.2',
        #                       '/home/suraj/resouces/book_holly/SoundofBlue_MASTER_1000.txt'],
        # 'Kingdom of Simplicity': ['Holly Lynn Payne', 'HKCaPgAACAAJ', '0982279779', '9780982279779',
        #                           'Skywriter Books; First edition (July 3, 2009)', '296', '4.8',
        #                           '/home/suraj/resouces/book_holly/KOS_1000.txt'],
        # 'DAMASCENA the tale of roses and Rumi': ['Holly Lynn Payne', '3ZH6oAEACAAJ', '0982279744',
        #                                          '9780982279748', 'Skywriter Books (June 1, 2014)', '348', '4.9',
        #                                          '/home/suraj/resouces/book_holly/DAMA_MASTER_1000.txt'],
        # "The virgin's knot": ['Holly Lynn Payne', 'B00658MH7K', 'B00658MH7K', '',
        #                       ' Skywriter Books; 1 edition (November 5, 2011)', '320', '3.7',
        #                       '/home/suraj/resouces/book_holly/vknot_1000.txt']
        # "The Bullet": ['Joaquin Lowe', 'isbnnotknown1', '', '', '', '260', '',
        #                '/home/suraj/resouces/book_holly/The_Bullet_1000.txt'],
        # "In her own sweet time": ['Rachel Lehmann-Haupt', 'isbnnotknown2', '', '', '', '387', '',
        #                           '/home/suraj/resouces/book_holly/In_her_sweet_time_1000.txt'],
        # "Sea Legs And Fish Nets": ['Maria Finn', 'isbnnotknown3', '', '', '', '231', '',
        #                            '/home/suraj/resouces/new_books_1000/isbnnotknown3.txt'],
        # "Embraceable Me: Lessons in Heartbreak, Desire, and the Argentine Tango": ['Maria Finn', 'isbnnotknown4', '',
        #                                                                            '', '', '36', '',
        #                                                                            '/home/suraj/resouces/new_books_1000/isbnnotknown4.txt'],
         "Sugarland: A Jazz Age Mystery": ['Martha Conway', '099161853X', '099161853X',
                                                                                   '9780991618538', 'Noontime Books; 1 edition (May 7, 2016)', '314', '',
                                                                                   '/home/suraj/resouces/remaining/099161853X_099161853X.txt']

    }

    for book, value in mapping.iteritems():
        content = read_book(value[-1], encoding='utf-8')
        Authors(book_id=value[1], title=book, slug='-'.join(book.split()), authors=[value[0]], isbn_10=value[2],
                isbn_13=value[3], publisher=value[4], pages=int(value[5]), content=content, is_active=False).save()

    print ('Done')


def upate_authors_book():
    base_dir = '/home/suraj/resouces/rejoshbooks/1000_sent'
    book_to_update = ['3ZH6oAEACAAJ_0982279744.txt',
                      'HKCaPgAACAAJ_0982279779.txt',
                      'isbnnotknown.txt',
                      'yiIrAQAAIAAJ_0982279752.txt',
                      '8st8cSkyGt8C_0345502183.txt',
                      'VgLKYawnwHgC_0345493419.txt',

                      '11231.txt',
                      '1342.txt',
                      '15492.txt',
                      '19942.txt',
                      '2852.txt',
                      '33.txt',
                      '5200.txt',
                      '541.txt',
                      '76.txt'

                      ]

    for book_id in book_to_update:
        content = read_book(os.path.join(base_dir, book_id), encoding='utf-8')
        Book.objects(book_id=book_id.replace('.txt', '').split('_')[0]).update_one(set__content=content)
        print ('Done', book_id)


# def update_google_book():
#     data_path='/home/booxby/PycharmProjects/Skywriterx/data'
#
#     for book in Book.objects(is_active=True, _cls='Book.GoogleBook'):
#         try:
#             isbn_10=int(book.isbn_10)
#         except Exception as e:
#             isbn_10=book.isbn_10
#         out_name=book.book_id+'.txt' if not book.isbn_10 else book.book_id+'_'+str(isbn_10)+'.txt'
#         content = read_book(os.path.join(data_path,out_name), encoding='utf-8')
#
#
#         content = re.sub('_+', '', content, re.UNICODE)
#
#         # content=re.sub(r"[[u'\u4e00'-u'\u9fff']+",' ', content)
#         # content = WIERED_UNICODE.sub('', content)
#         book.update(set__content=content)
#         print ('Done', book.book_id)


def update_google_book():
    # data_path='/home/booxby/PycharmProjects/Skywriterx/data'

    for book in Book.objects(is_active=True, _cls='Book.GoogleBook'):
        # try:
        #     isbn_10=int(book.isbn_10)
        # except Exception as e:
        #     isbn_10=book.isbn_10
        # out_name=book.book_id+'.txt' if not book.isbn_10 else book.book_id+'_'+str(isbn_10)+'.txt'

        content = book.content
        # content= re.sub(ur'[\u4e00-\u9fff]+','',content)
        # content= re.sub(ur'ziч','',content)

        content = re.sub('_+', '', content)

        # content=re.sub(r"[[u'\u4e00'-u'\u9fff']+",' ', content)
        # content = WIERED_UNICODE.sub('', content)
        book.update(set__content=content)
        print ('Done', book.book_id)


def generate_synthetic_tags():
    """
    This method generates synthetic tags from its similar books
    Steps:
    1. For each book, get top 15 similar books
    2. Generate 3 sets of tags (From 5, 10, 15 similar books)
    3. In each set,
       > List all the distinct tags appeared in its similar books.
       > Count how many books have this particular tag as frequency
       > Sort then based on this frequency count.
    4. Store top 15 tags in the database
    5. Tags from 5 similar books is stored in synthetic_tags_5 field
       > Tags from 10 similar books is stored in synthetic_tags_10 field
       > Tags from 15 similar books is stored in synthetic_tags_15 field
    :return:
    """

    def get_tags_from_n_similar_books(most_similar, n):
        tags_and_count = {}

        # Get Tags from top 5 similar books
        for similar_book_id, similar_book_scores in most_similar[1:n+1].iterrows():

            tags_with_count = Book.objects.get(book_id=similar_book_id)['tags']

            for tag_count in tags_with_count:
                tag, count = tag_count.split('(')
                tags_and_count[tag] = tags_and_count.get(tag, 0) + 1

        sorted_tags = sorted(tags_and_count.items(), key=operator.itemgetter(1), reverse=True)
        sorted_tags_formatted = [k+'('+str(v)+')' for k,v in sorted_tags]
        return sorted_tags_formatted[:min(15, len(sorted_tags))]

    from bookweb.engine import Recommendation
    from bookweb.engine import FeaturesSimilarity
    from flask import current_app

    app = current_app._get_current_object()
    features = app.config['FEATURES']
    recommendation = Recommendation()

    counter = 0
    for book in Book.objects(is_active=True):
        book_id = book.book_id
        counter += 1

        feature_layer = FeaturesSimilarity(book, features)
        recommendation.add_layer(feature_layer)
        # most_similar, least_similar = get_n_similar_books(id, 10, features)
        most_similar, least_similar = recommendation.get_n_similar_books(16, kernel='cosine')

        print('---------------- 5 books ------------')
        book.update(set__synthetic_tags_5=get_tags_from_n_similar_books(most_similar, 5))
        book.update(set__synthetic_tags_10=get_tags_from_n_similar_books(most_similar, 10))
        book.update(set__synthetic_tags_15=get_tags_from_n_similar_books(most_similar, 15))

        print('Tags generated for {} books, Name {}, Id {}'.format(counter, book.title, book_id))



def dump_data(output_dir):
    for book in Book.objects(is_active=True):
        out_name = book.book_id + '.txt' if not book.isbn_10 else book.book_id + '_' + book.isbn_10 + '.txt'
        with codecs.open(os.path.join(output_dir, out_name), 'w', encoding='utf-8') as f_out:
            content = book.content.replace('\n', ' ').replace('\r', '')

            f_out.write(content)


if __name__ == '__main__':
    # Google_Books = '/home/booxby/PycharmProjects/Skywriterx/book_info'
    #
    # save_googlebook_to_db(Google_Books)
    # save_gutenberg_books()
    save_authors_book()

def crawl():
    import time
    import requests
    from BeautifulSoup import BeautifulSoup
    HEADERS = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}

    proxies_list = ["128.199.109.241:8080","113.53.230.195:3128","125.141.200.53:80","125.141.200.14:80","128.199.200.112:138","149.56.123.99:3128","128.199.200.112:80","125.141.200.39:80","134.213.29.202:4444"]

    proxies = {'https': random.choice(proxies_list)}
    #HEADERS = {"User-Agent": "FireFox/2.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 FireFox/19.0"}

    datafile = open('/home/ritual/Desktop/ml-master/book_eacl/finalFile', 'r')

    target2 = open('eRRORfIL', 'w')
    for line in datafile:

        data = line.split(":")
        bookId = data[0]
        author = data[1].strip()
        title = Book.objects.get(book_id=bookId)['title'].strip()
        if title.endswith('LP') or title.endswith('Lp'):
            title = title[:-3].strip()
        title = title.replace(' ', '+')

        if author[0] == 'c':
            author = author[1:]
        author = author.replace(' ', '+')
        addressOfFile = '/home/ritual/Desktop/ml-master/amazon/' + title

        target = open(addressOfFile, 'w')
        query = 'http://www.amazon.com/s/ref=nb_sb_noss?url=search-alias%3Dstripbooks&field-keywords=' + title + '+' + author
        target.write(query)
        target.write("\n")
        target.write("\n")
        print query

        status = -1
        status2 = -1

        response = None
        while status != '200':  # Request may fail. So we will try until it becomes successful
            try:
                response = requests.get(query, headers=HEADERS)
            except:
                time.sleep(1)
                response = requests.get(query, headers=HEADERS)
            status = str(response.status_code)
            if (status == '503'):
                time.sleep(600)

        content = response.content
        print response.content

        soup = BeautifulSoup(content)

        result_1 = soup.find('li', {'id': 'result_0'})  # First result
        print result_1

        #a = result_1.find('a', {'class': 'a-link-normal s-access-detail-page  a-text-normal'})
        a2 = result_1.find('a', {'class': 'a-link-normal a-text-normal'})
        query2 = a2.get('href')
        print query2

        response2 = None
        while status2 != '200':  # Request may fail. So we will try until it becomes successful
            try:
                time.sleep(600)
                response2 = requests.get(query2, headers=HEADERS)
            except:
                time.sleep(1)
                response2 = requests.get(query2, headers=HEADERS)
            status2 = str(response2.status_code)
            print(status2)
            if (status2 == '503'):
                time.sleep(600)
        content2 = response2.content
        soup2 = BeautifulSoup(content2)
        numberOfComents = len(soup2.findAll('div', {'class': 'a-row review-data'}))
        for j in range(numberOfComents):
            result_2 = soup2.findAll('div', {'class': 'a-row review-data'})[j].text.strip()

            target.write(result_2.encode('utf-8'))
            target.write("\n")
            target.write("\n")
            target.write("\n")

        time.sleep(600)
            # print result_2
