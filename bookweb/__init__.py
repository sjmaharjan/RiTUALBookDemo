__author__ = 'suraj'

from flask import Response
from flask_admin.contrib.mongoengine import ModelView

from werkzeug.exceptions import HTTPException

from werkzeug.utils import redirect

from flask import Flask
from flask import render_template
from flask_admin import AdminIndexView
from flask_admin import expose

from flask_basicauth import BasicAuth
from flask_moment import Moment
from flask_mongoengine import MongoEngine, MongoEngineSessionInterface
from flask_bootstrap import Bootstrap
from flask import request, url_for
from config import config
from bookweb import fvs_loader
from flask_debugtoolbar import DebugToolbarExtension
from celery import Celery
from config import Config
from flask_admin import Admin
from flask import Flask
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)

basic_auth = BasicAuth(app)
app.config['BASIC_AUTH_USERNAME'] = 'xxxxxx'
app.config['BASIC_AUTH_PASSWORD'] = 'yyyyyy'
bootstrap = Bootstrap()
db = MongoEngine()
moment = Moment()
fvs = fvs_loader.FeatureInitializer()
explangemb=fvs_loader.ExperientialLanguageEmbeddingInitializer()
#toolbar = DebugToolbarExtension()
celery = Celery(__name__, backend=Config.CELERY_BACKEND_URL, broker=Config.CELERY_BROKER_URL)

from .admin import   GoogleBookModelView, GutenbergBookModelView, AuthorsBookModelView, BuildModelView, Index


def url_for_other_page(page):
    args = request.view_args.copy()
    args['page'] = page
    return url_for(request.endpoint, **args)


def register_blueprints(app):
    # Prevents circular imports
    from bookweb.books import books
    from bookweb.recommendation import recommendations
    from bookweb.comments import comments
    from bookweb.statistic import statistic
    #from booxby.admin import admin
    from bookweb.feature_viewer import features_view


    app.register_blueprint(books)
    app.register_blueprint(recommendations)
    app.register_blueprint(comments)
    app.register_blueprint(features_view)
    app.register_blueprint(statistic)
   # app.register_blueprint(admin)

class AuthException(HTTPException):
    def __init__(self, message):
        print "errrrr"

class ModelView(ModelView):
    def is_accessible(self):
        if not basic_auth.authenticate():
            AuthException('Not authenticated.')
        else:
            return True

    def inaccessible_callback(self, name, **kwargs):
        return redirect(basic_auth.challenge())


def create_app(config_name):
    app = Flask(__name__)
    #admin = Admin(name="Booxby")


    admin = Admin(template_mode='bootstrap3',base_template='admin/master.html')
    admin.add_view(Index(name='test', endpoint='test'))

    app.config.from_object(config[config_name])

    config[config_name].init_app(app)
    bootstrap.init_app(app)
    db.init_app(app)
    moment.init_app(app)
    fvs.init_app(app)
    explangemb.init_app(app)
    celery.conf.update(app.config)



    admin.init_app(app)



    from bookweb.models import Book, GoogleBook, GutenbergBook, Authors


    from flask_admin.contrib.fileadmin import FileAdmin


    admin.add_view(BuildModelView(name='Build',endpoint='build'))
    #admin.add_view(MenuModelView(name='Home2', menu_icon_type='', menu_icon_value='glyphicon-home'))

    basic_auth = BasicAuth(app)
    admin.add_view(

        GoogleBookModelView(GoogleBook)

    )

    admin.add_view(

       GutenbergBookModelView(GutenbergBook,basic_auth)

    )
    admin.add_view(

        AuthorsBookModelView(Authors)

    )


    path = app.config['UPLOAD_FOLDER']
    admin.add_view(FileAdmin(path, name='Uploaded Books'))
    app.jinja_env.globals['url_for_other_page'] = url_for_other_page
    app.session_interface = MongoEngineSessionInterface(db)
    app.config['BASIC_AUTH_USERNAME'] = 'admin_ritual'
    app.config['BASIC_AUTH_PASSWORD'] = 'Random1368_Test'
    #app.config['BASIC_AUTH_FORCE'] = True
    basic_auth = BasicAuth(app)
    admin_view=admin.index_view

    @app.route('/secret')
    @basic_auth.required
    def secret_view():
        return render_template("admin/index.html")
    register_blueprints(app)
    # here



        # here
    # from booxby import  websockets
    return app




    #
    # if __name__ == '__main__':
    #     app.run()
