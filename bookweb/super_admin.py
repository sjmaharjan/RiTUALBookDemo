__author__ = 'suraj'

from flask import Blueprint, request, redirect, render_template, url_for, send_from_directory, flash, jsonify
from flask.views import MethodView
from flask_mongoengine.wtf import model_form
from flask_wtf import Form
from werkzeug.utils import secure_filename
import os
from wtforms import BooleanField, StringField, TextAreaField, IntegerField, FileField
from bookweb.models import Book, GoogleBook, GutenbergBook, Authors, BookStatus
from flask import current_app
from bookweb import celery
from celery import chord, group
import codecs
from celery import chain
import re

admin = Blueprint('admin', __name__, template_folder='templates')


def is_google_book_id(id):
    try:
        from scraper.google_books import download_meta_info
        return download_meta_info(id)
    except:
        return None


def generate_id():
    books_without_isbn = Book.objects(book_id__startswith='isbnnotknown')
    vals = [int(book.book_id.replace('isbnnotknown', '')) for book in books_without_isbn]
    if not vals:
        max_vals = 0
    else:
        max_vals = max(vals)
    next_id = max_vals + 1
    return 'isbnnotknown' + str(next_id)


class BookForm(Form):
    book_id = StringField('Book ID')
    title = StringField('Title')
    isbn_10 = StringField('ISBN 10')
    isbn_13 = StringField('ISBN 13')
    authors = StringField('Authors')
    skip_pages = IntegerField('Skip Pages', default=0)
    book = FileField("Book")


# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    app = current_app._get_current_object()
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config[
        'ALLOWED_EXTENSIONS']


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



class List(MethodView):
    cls = Book

    def get(self):
        books = self.cls.objects.all()
        return render_template('admin/list.html', books=books)


class Detail(MethodView):
    # Map book types to models
    class_map = {
        'GoogleBook': GoogleBook,
        'Authors': Authors,
        'GutenbergBook': GutenbergBook,

    }

    def get_context(self, id=None, slug=None):

        if id:

            book = Book.objects.get_or_404(book_id=id)
            # Handle old posts types as well
            cls = book.__class__ if book.__class__ != Book else Authors
            form_cls = model_form(cls,
                                  exclude=('comments', 'tags', 'genre', 'authors', 'book_source', 'slug'))
            if request.method == 'POST':
                form = form_cls(request.form, inital=book._data)
            else:
                form = form_cls(obj=book)
        else:
            # Determine which book type we need
            cls = self.class_map.get(request.args.get('type', 'Authors'))
            book = cls()
            form = BookForm(request.form, csrf_enabled=False)
        context = {
            "book": book,
            "form": form,
            "create": id is None
        }
        return context

    def get(self, id, slug):
        context = self.get_context(id, slug)
        return render_template('admin/detail.html', **context)

    def post(self, id=None, slug=None):
        msg = ''

        app = current_app._get_current_object()
        context = self.get_context(id, slug)
        form = context.get('form')

        if id:
            if form.validate():
                book = context.get('book')
                form.populate_obj(book)
                book.save()
                flash("Book Updated")
                return redirect(url_for('admin.books'))
        else:
            if form.validate_on_submit():
                book = context.get('book')
                book_id = form.book_id.data
                isbn_13 = form.isbn_13.data
                isbn_10 = form.isbn_10.data
                authors = form.authors.data
                title = form.title.data
                skip_pages = form.skip_pages.data
                file = request.files[form.book.name]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                content = get_book_content(os.path.join(app.config['UPLOAD_FOLDER'], filename), skip_pages)
                if content == '':
                    msg = 'Book Content is empty. Check the uploaded file'
                else:
                    content = book.get_n_sentences(content)  # get 10000 sentences
                google_book = is_google_book_id(book_id)
                if google_book:
                    from utils import google_book_mapping
                    data = google_book_mapping(google_book)
                    book.populate_obj(status=BookStatus.UNPROCESSED, content=content,
                                      book_file_name=os.path.join(app.config['UPLOAD_FOLDER'], filename), **data)
                    book.save()
                    msg = 'Book Id matched google book id, so overriding user input.'
                else:
                    msg = ''
                    if not book_id:
                        book_id = generate_id()

                    book.populate_obj(book_id=book_id, isbn_13=isbn_13, isbn_10=isbn_10, authors=authors.split(','),
                                      title=title, content=content,
                                      slug="-".join(title.replace('(', '').replace(')', '').replace('/', '').split()),
                                      status=BookStatus.UNPROCESSED,
                                      book_file_name=os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    book.save()

                flash('Book Added Successfully. ' + msg)

                return redirect(url_for('admin.books'))
        return render_template('admin/detail.html', **context)


class DashBoardListView(MethodView):
    def get(self):
        stats = {
            'Active_books': Book.objects(is_active=True).count(),
            'Removed_books': Book.objects(is_active=False).count(),
            'Unprocessed_books': Book.objects(status=BookStatus.UNPROCESSED).count(),
        }

        return render_template('admin/dashboard.html', stats=stats)


@celery.task(bind=True)
def parse_and_build_model(self):
    from parsers.tasks import run_stanford_parser, run_sentic_parser
    from dump_vectors import build_model
    app = current_app._get_current_object()
    files_to_process = []
    for book in Book.objects(status=BookStatus.UNPROCESSED):
        fname = book.book_id + '.txt' if not book.isbn_10 else book.book_id + '_' + book.isbn_10 + '.txt'
        with codecs.open(os.path.join(app.config['TEMP'], fname), 'w', encoding='utf-8') as f_out:
            content = book.content.replace('\n', ' ').replace('\r', '').replace('\x0C', '')
            content = re.sub(ur'[\u4e00-\u9fff]+', '', content)
            f_out.write(content)
        files_to_process.append(os.path.join(app.config['TEMP'], fname))
        print('Done writing files to tmp')

    self.update_state(state='PROGRESS',
                          meta={'current': len(files_to_process), 'total': len(files_to_process)*3+len(app.config['FEATURES']),
                                'status': "Done Dumping content to files"})

    res = chord(group(chain(run_stanford_parser.s(os.path.join(app.config['TEMP'], file)), run_sentic_parser.s()) for
         file in files_to_process))(group( build_model.si(feature) for feature in app.config['FEATURES'] )).apply_async()
    # if res.ready():
    #     if res.successful():
    #         self.update_state(state='PROGRESS',
    #                       meta={'current': len(files_to_process)*3, 'total': len(files_to_process)*3+len(app.config['FEATURES']),
    #                             'status': "Done Parsing"})
    #
    #         model=group(build_model.s(features) for features in app.config['FEATURES']).apply_async()
    #         if model.ready():
    #             if model.successful():
    #                 self.update_state(state='Done',
    #                       meta={'current': len(files_to_process)*3+len(app.config['FEATURES']), 'total': len(files_to_process)*3+len(app.config['FEATURES']),
    #                             'status': "Done Building Model"})
    #             else:
    #                 self.update_state(state='Failure',
    #                       meta={'current': len(files_to_process)*3, 'total': len(files_to_process)*3+len(app.config['FEATURES']),
    #                             'status': "Failed Building Model"})



    # ( group(build_model.si(features) for features in app.config['FEATURES'])).apply_async()
    # print('Running Model')
    # for features in app.config['FEATURES']:
    #     build_model.s(features).apply_async()


    return {'current': len(files_to_process)*3+len(app.config['FEATURES']), 'total': len(files_to_process)*3+len(app.config['FEATURES']), 'status': 'Task completed!', 'result': 42}


class ModelListView(MethodView):
    def get(self):
        unprocessed_books = Book.objects(status=BookStatus.UNPROCESSED)
        return render_template('admin/model.html', books=unprocessed_books)

    def post(self):
        if request.method == 'POST':  # and form.validate():

            task = parse_and_build_model.apply_async()
            return jsonify({}), 202, {'Location': url_for('admin.taskstatus',
                                                          task_id=task.id)}


@admin.route('/model/status/<task_id>')
def taskstatus(task_id):
    task = parse_and_build_model.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


# Register the urls
admin.add_url_rule('/admin/', view_func=DashBoardListView.as_view('index'))
admin.add_url_rule('/admin/books', view_func=List.as_view('books'))
admin.add_url_rule('/admin/books/create/', defaults={'id': None, 'slug': None}, view_func=Detail.as_view('create'),
                   methods=['POST', 'GET'])
admin.add_url_rule('/admin/<id>/<slug>/', view_func=Detail.as_view('edit'))
admin.add_url_rule('/admin/model/', view_func=ModelListView.as_view('model'))
admin.add_url_rule('/admin/model/update', view_func=ModelListView.as_view('model_update'))
