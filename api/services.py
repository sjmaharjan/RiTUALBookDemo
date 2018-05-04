__author__ = 'suraj'

from bookweb.models import Book
from bookweb import books

'''
Returns json tags. Tags meaning "insperational, motvational, etc.
'''
@books.route('/recommendation/<id>/tags')
def experiential_language_tags(id):
        book = Book.objects.get_or_404(book_id=id)
        tags=book.get_experiential_languages()
        return flask.jsonify(experiential_language_tags=tags)


'''
Returns json recommendation books based on seed
'''
@books.route('/recommendation/<id>/')
def experiential_language_tags(id):
        book = Book.objects.get_or_404(book_id=id)
        tags=book.get_experiential_languages()
        return flask.jsonify(experiential_language_tags=tags)