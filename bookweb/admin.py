__author__ = 'suraj'

from flask import render_template
from flask.views import MethodView

from flask_admin import BaseView, expose, form
from flask_admin.babel import gettext
from flask_admin.contrib.mongoengine import ModelView
from flask_admin.helpers import get_redirect_target
from flask_admin.model.helpers import get_mdict_item_or_list
from bookweb.models import BookStatus, Book
from config import Config
from flask import flash, jsonify, request, url_for
import os, codecs
from flask import current_app, redirect
import re
from bookweb import celery
from celery import chord, group, chain
import time
from wtforms import SelectField,TextAreaField
from flask_wtf import Form
from flask_admin.helpers import get_redirect_target
from flask_httpauth import HTTPBasicAuth
from functools import wraps
from flask import request, Response
from flask_basicauth import BasicAuth


file_path = Config.UPLOAD_FOLDER

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        auth = request.authorization
        if not auth or not (auth.username == 'yyyyyyyy' and auth.password == 'xxxxxxxxx'):
            return Response(
            'Could not verify your access level for that URL.\n'
            'You have to login with proper credentials', 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'})

        return f(*args, **kwargs)

    return decorated




def get_file_converter(fname):
    """ Factory to get the file to text converter
        The converter is choosen based on the extension


    """
    from helpers.file_converters import pdf2txt, doc2txt, txt2txt
    converters = dict(
        pdf=lambda skip: pdf2txt(fname, skip),
        docx=lambda skip: doc2txt(fname, skip),
        txt=lambda skip: txt2txt(fname, skip),

    )
    extension = os.path.basename(fname).rsplit('.', 1)[1]
    return converters.get(extension, None)


def get_book_content(fname, skip=None):
    converter = get_file_converter(fname)
    if converter:
        return converter(skip)
    else:
        return ""


def is_google_book_id(id):
    try:
        from scraper.google_books import download_meta_info
        return download_meta_info(id)
    except:
        return None


def generate_id():
    from bookweb.models import Book
    books_without_isbn = Book.objects(book_id__startswith='isbnnotknown')
    vals=[]
    for book in books_without_isbn:
	num=book.book_id.replace('isbnnotknown', '')
	if num:
		num= int (num)
		vals.append(num) 
    if not vals:
        max_vals = 0
    else:
        max_vals = max(vals)
    next_id = max_vals + 1
    return 'isbnnotknown' + str(next_id)




class EditContentForm(Form):           
    content = TextAreaField('content') 



class BookModelView(ModelView):
    can_export = True

    column_filters = ('title', 'is_active')
    column_exclude_list = ['content', 'comments', 'slug', 'book_id', 'book_file_name', 'book_source', '_cls', 'url',
                           'publisher', 'published_date', 'tags', 'skip_pages', 'cover_image', 'description',
                           'self_link', 'page_count','status']

    column_searchable_list = ['title', 'content']
    column_editable_list = ['title']

    form_columns = (
        'book_id', 'isbn_10', 'isbn_13', 'title', 'genre', 'authors', 'publisher', 'published_date', 'tags',
        'page_count',
        'book_file_name', 'content', 'status')

    form_excluded_columns = (
        'slug', 'book_source', 'self_link', 'cover_image', 'description', 'is_active', 'content', 'url')

    # Override form field to use Flask-Admin FileUploadField
    form_overrides = {
        'book_file_name': form.FileUploadField,
        'status': SelectField
    }

    # Pass additional parameters to 'path' to FileUploadField constructor
    form_args = {

        'book_file_name': {
            'label': 'Book',
            'base_path': file_path,
            'allow_overwrite': False
        },

        'status': {
            'choices': [
                (BookStatus.INACTIVE, 'Inactive'),
                (BookStatus.UNPROCESSED, 'UnProcessed'),
                (BookStatus.ACTIVE, 'Active')
            ],
            'coerce': int

        }
    }


    #

    def on_model_change(self, form, model, is_created):
        if is_created:
            msg = ''
            if model.book_file_name:
                content = get_book_content(os.path.join(file_path, model.book_file_name), model.skip_pages)
            else:
                content = model.content
            if content == '':
                msg = 'Book Content is empty. Check the uploaded file'

            google_book = is_google_book_id(model.book_id)

            if google_book:
                from utils import google_book_mapping
                from bookweb.models import BookStatus
                data = google_book_mapping(google_book)
                model.populate_obj(content=content, **data)

                msg += 'Book Id matched google book id, so overriding user input.'
            else:
                if not model.book_id:
                    book_id = generate_id()
                #model.book_id = book_id
                model.slug = "-".join(model.title.replace('(', '').replace(')', '').replace('/', '').split())
                model.content = content
            flash(msg)

        else:

            pass


    def get_save_return_url(self, model, is_created=False):
        if is_created:
            return self.get_url('.create_continue', id=model.id)
        return self.get_url('.index_view')

    @requires_auth
    @expose('/continue/<id>', methods=('GET', 'POST'))
    def create_continue(self, id):

        return_url = get_redirect_target() or self.get_url('.index_view')
        model = Book.objects.get_or_404(id=id)

        if model is None:
            flash(gettext('Record does not exist.'))
            return redirect(return_url)

        if request.method == 'POST':
            form = EditContentForm(request.form)

            content = form.content.data
            content= model.get_n_sentences(content)
            model.update(set__content=content)
            flash(gettext('Record was successfully saved.'))
            return redirect(self.get_url('.index_view'))

        if request.method == 'GET':
            form = EditContentForm(content=model.content, csrf_enabled=False)

        template = 'admin/continue.html'

        return self.render(template,
                           model=model,
                           form=form,
                           )


    def get_save_return_url(self, model, is_created=False):                                      
       if is_created:                                                                           
           return self.get_url('.create_continue', id=model.id)                                 
       return self.get_url('.index_view')


    @expose('/continue/<id>', methods=('GET', 'POST'))                                           
    def create_continue(self, id):                                                               
                                                                                                
       return_url = get_redirect_target() or self.get_url('.index_view')                        
       model = Book.objects.get_or_404(id=id)                                                   
                                                                                                
       if model is None:                                                                        
           flash(gettext('Record does not exist.'))                                             
           return redirect(return_url)                                                          
                                                                                                
       if request.method == 'POST':                                                             
           form = EditContentForm(request.form)                                                 
                                                                                                
           content = form.content.data                                                          
           content= model.get_n_sentences(content)                                              
           model.update(set__content=content)                                                   
           flash(gettext('Record was successfully saved.'))                                     
           return redirect(self.get_url('.index_view'))                                         
                                                                                                
       if request.method == 'GET':                                                              
           form = EditContentForm(content=model.content, csrf_enabled=False)                    
                                                                                                
       template = 'admin/continue.html'                                                         
                                                                                                
       return self.render(template,                                                             
                          model=model,                                                          
                          form=form,                                                            
                          )                                                                     
                                                                                                
from werkzeug.exceptions import HTTPException

class AuthException(HTTPException):
    def __init__(self, message):
        super(AuthException, self).__init__(message, Response(
            "You could not be authenticated. Please refresh the page.", 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'}
        ))


class GutenbergBookModelView(BookModelView):
    basic_auth = BasicAuth()
    def __init__(self,BookModelView, basic_auth):
        self.basic_auth = basic_auth
        super(GutenbergBookModelView, self).__init__(BookModelView)

    def is_accessible(self):
        if not self.basic_auth.authenticate():
            raise AuthException('Not authenticated.')
        else:
            return True

    def inaccessible_callback(self, name, **kwargs):
        return redirect(self.basic_auth.challenge())




class GoogleBookModelView(BookModelView):
        pass


class AuthorsBookModelView(BookModelView):
        pass


@celery.task
def dump_to_tmp(id, dump_dir):
    # print ("%%%%%%%%%%%%%%%%%%%%%%%%%")
    # print (dump_dir)
    # print ("%%%%%%%%%%%%%%%%%%%%%%%%%")
    from bookweb.models import Book
    book = Book.objects.get(book_id=id)
    fname = book.book_id + '.txt' if not book.isbn_10 else book.book_id + '_' + book.isbn_10 + '.txt'
    if not os.path.exists(os.path.join(dump_dir, fname)):
        with codecs.open(os.path.join(dump_dir, fname), 'w', encoding='utf-8') as f_out:
            content = book.content.replace('\n', ' ').replace('\r', '').replace('\x0C', '')
            content = re.sub(r'[\u4e00-\u9fff]+', '', content)
            f_out.write(content)
    return os.path.join(dump_dir, fname)


class BuildModelView(BaseView):
    @expose('/')
    #@requires_auth
    def index(self):
        from bookweb.models import Book, BookStatus
        unprocessed_books = Book.objects(status=BookStatus.UNPROCESSED)
        return self.render('admin/model.html', books=unprocessed_books)

    @expose('/update', methods=('POST',))
    def update(self):

        if request.method == 'POST':
            from dump_vectors import build_model
            app = current_app._get_current_object()
            for f in os.listdir(app.config['STAGGED_VECTORS']):
                os.remove(os.path.join(app.config['STAGGED_VECTORS'], f))
            jobs = group(build_model.s(feature) for feature in app.config['FEATURES'])
            self._feature_task = jobs.apply_async()

            return jsonify({}), 202, {'Location': url_for('.taskstatus')}

    @expose('/update/parse', methods=('POST',))
    def parse(self):

        if request.method == 'POST':
            from parsers.tasks import run_stanford_parser, run_sentic_parser
            app = current_app._get_current_object()
            dump_dir = app.config['TEMP']
            jobs = group((dump_to_tmp.s(book.book_id, dump_dir)|run_stanford_parser.s()|run_sentic_parser.s())
                         for book in Book.objects(status=BookStatus.UNPROCESSED))
            self._parse_task = jobs.apply_async()
            return jsonify({}), 202, {'Location': url_for('.parsestatus')}

    @expose('/taskstatus/parse', methods=('GET',))
    def parsestatus(self):
        task = self._parse_task
        if task.waiting():
            response = {
                'state': 'PROGRESS',
                'completed': task.completed_count(),
                'tasks': len(task.children),
                'status': 'Pending...'
            }
        elif task.successful():
            response = {
                'state': 'SUCCESS',
                'status': 'Done',
                'result': "Task Complete"
            }

        else:
            # something went wrong in the background job
            response = {
                'state': 'FAILURE',
                'status': 'SOMETHING WENT WRONG',  # this is the exception raised
            }
        return jsonify(response)

    @expose('/taskstatus', methods=('GET',))
    def taskstatus(self):
        task = self._feature_task
        if task.waiting():
            response = {
                'state': 'PROGRESS',
                'completed': task.completed_count(),
                'tasks': len(task.children),
                'status': 'Pending...'
            }
        elif task.successful():
            response = {
                'state': 'SUCCESS',
                'status': 'Done',
                'result': "Task Complete"
            }

        else:
            # something went wrong in the background job
            response = {
                'state': 'FAILURE',
                'status': 'SOMETHING WENT WRONG',  # this is the exception raised
            }
        return jsonify(response)

    @expose('/update/activate/<id>', methods=('GET', 'POST'))
    def activate(self, id):
        book = Book.objects.get_or_404(book_id=id)
        book.update(set__is_active=True)
        book.update(set__status=BookStatus.ACTIVE)
        flash('Book is now active.')
        return redirect(url_for('.index'))

    @expose('/update/activateall', methods=('GET', 'POST'))
    def activateall(self):

        for book in Book.objects(status=BookStatus.UNPROCESSED):
            book.update(set__is_active=True)
            book.update(set__status=BookStatus.ACTIVE)
        flash('All processed books are now active.')
        return redirect(url_for('.index'))



class Index(BaseView):

    @expose('/')
    #@requires_auth
    def index(self):
        return self.render("admin/index.html")
