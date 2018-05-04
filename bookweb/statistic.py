from flask import Blueprint, request,redirect, render_template, url_for, flash
from flask.views import MethodView
from bookweb.utils import get_comments_count
from bookweb.models import Book, Comment
from flask_mongoengine.wtf.orm import model_form
from flask_httpauth import HTTPBasicAuth
from functools import wraps
from flask import request, Response

statistic = Blueprint('statistic', __name__, template_folder='templates')


class ListView(MethodView):
    def get(self):

        book_source = Book.objects(is_active=True).aggregate(
                                                             {"$group": {"_id": "$book_condition", "count": {"$sum": 1}}})
        book_source_g = Book.objects(is_active=True).aggregate({"$unwind": "$genre"},
                                                             {"$group": {"_id": "$genre", "count": {"$sum": 1}}})
        return render_template('statistic/list.html', book_s=book_source, book_s_g=book_source_g)



statistic.add_url_rule('/statistic', view_func=ListView.as_view('list'), methods=['GET', ])

