
__author__ = 'suraj'

from flask import Flask
from flask_moment import Moment
from flask_mongoengine import MongoEngine, MongoEngineSessionInterface
from flask_bootstrap import Bootstrap
from flask import request, url_for
from config import config
from bookweb import fvs_loader
from celery import Celery
from config import Config


app = Flask(__name__)

bootstrap = Bootstrap()
db = MongoEngine()
moment = Moment()
fvs = fvs_loader.FeatureInitializer()
explangemb=fvs_loader.ExperientialLanguageEmbeddingInitializer()
celery = Celery(__name__, backend=Config.CELERY_BACKEND_URL, broker=Config.CELERY_BROKER_URL)




def url_for_other_page(page):
    args = request.view_args.copy()
    args['page'] = page
    return url_for(request.endpoint, **args)


def register_blueprints(app):
    # Prevents circular imports
    from bookweb.books import books
    from bookweb.recommendation import recommendations
    from bookweb.comments import comments
    # from booxby.admin import  admin
    from bookweb.feature_viewer import features_view

    app.register_blueprint(books)
    app.register_blueprint(recommendations)
    app.register_blueprint(comments)
    app.register_blueprint(features_view)
    # app.register_blueprint(admin)


def create_app_api(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    bootstrap.init_app(app)
    db.init_app(app)
    moment.init_app(app)
    fvs.init_app(app)
    explangemb.init_app(app)
    celery.conf.update(app.config)



    from bookweb.models import Book, GoogleBook, GutenbergBook, Authors
    from flask_admin.contrib.fileadmin import FileAdmin



    app.jinja_env.globals['url_for_other_page'] = url_for_other_page
    app.session_interface = MongoEngineSessionInterface(db)
    register_blueprints(app)

    # from booxby import  websockets
    return app
