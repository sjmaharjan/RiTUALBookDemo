__author__ = 'suraj'

from flask import Blueprint, request,redirect, render_template, url_for, flash
from flask.views import MethodView
from bookweb.utils import get_comments_count
from bookweb.models import Book, Comment
from flask_mongoengine.wtf.orm import model_form






comments = Blueprint('comments', __name__, template_folder='templates')


class ListView(MethodView):
    def get(self):

        books = Book.objects(comments__exists=True)
        # for book in books:
        #     print book.title
        comments=[]
        for book in books:
            for comment in book.comments:
                comment.book=book
                comments.append(comment)


        return render_template('comments/list.html', comments=comments)


class DetailView(MethodView):
    form = model_form(Comment, exclude=['comment_id', 'created_at'])

    def get_context(self, id, slug):
        book = Book.objects.get_or_404(book_id=id)

        form = self.form(request.form)

        context = {
            "book": book,
            "comments":book.comments,
            "form": form,
            "comment_count":get_comments_count()
        }
        return context

    def get(self,id,slug):
        context=self.get_context(id,slug)
        return render_template('comments/detail.html', **context)


    def post(self, id, slug):

        context = self.get_context(id, slug)
        form = context.get('form')

        if form.validate():
            comment = Comment()
            form.populate_obj(comment)

            book = context.get('book')
            comment.comment_id = len(book.comments)
            book.comments.append(comment)
            book.save()
            flash('Comment Saved.')

            return  redirect(url_for('comments.detail', id=id, slug=slug))
        flash('Could not save Comment.')

        return render_template('comments/detail.html', **context)



comments.add_url_rule('/comments', view_func=ListView.as_view('list'), methods=['GET', 'POST'])
comments.add_url_rule('/comments/<id>/<slug>', view_func=DetailView.as_view('detail'), methods=['GET', 'POST'])
