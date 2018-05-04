__author__ = 'suraj'

from flask import Blueprint,  render_template
from flask.views import MethodView
from bookweb.models import Book
from bookweb.engine import get_feature_values

features_view = Blueprint('features', __name__, template_folder='templates')




class FeaturesListView(MethodView):
    def get(self,feature_name,src_id,des_id=None):

        src = get_feature_values(feature_name, src_id)
        if des_id:
            des = get_feature_values(feature_name, des_id)
            return render_template('features/list.html', src_book=Book.objects.get(book_id=src_id),
                                   des_book=Book.objects.get(book_id=des_id), feature_values=zip(src, des))
        return render_template('features/list.html', src_book=Book.objects.get(book_id=src_id), feature_values=src)



list_view=FeaturesListView.as_view('list')
features_view.add_url_rule('/features/<feature_name>/<src_id>/<des_id>/', view_func=list_view,methods=['GET','POST'])
features_view.add_url_rule('/features/<feature_name>/<src_id>/', view_func=list_view,defaults={'des_id': None},methods=['GET','POST'])