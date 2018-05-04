
__author__ = 'suraj'

from flask import Blueprint, request,redirect, render_template, url_for, flash
from flask.views import MethodView
from bookweb.utils import get_comments_count
from bookweb.models import Book, Comment
from flask_mongoengine.wtf.orm import model_form
from flask_httpauth import HTTPBasicAuth
from functools import wraps
from flask import request, Response


recommendationsDetails = Blueprint('re', __name__, template_folder='templates')


class FeaturesDiagramView(MethodView):
    def get(self,my_plot):
        print "I am here"
        return render_template('recommendations/details.html', my_plot=my_plot)





reco_details = FeaturesDiagramView.as_view('details')
recommendationsDetails.add_url_rule('/recommendations/details/',view_func=reco_details, methods=['GET', ])