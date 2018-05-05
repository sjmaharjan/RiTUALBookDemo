# -*- coding: utf-8 -*-

# from gevent import monkey
# monkey.patch_all()

import os
import sys
import json
import codecs
import yaml

# Set the path
__author__ = 'suraj'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from flask_script import Manager, Command
from bookweb import create_app
from flask_script import Manager, Server, Command
from bookweb import create_app

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
manager = Manager(app)




# Turn on debugger by default and reloader
manager.add_command("runserver", Server(
    use_debugger=True,
    use_reloader=True,
    host='0.0.0.0',
    processes=100,
    #    port='9000'
))






if __name__ == "__main__":
    # insertgutenbergbooks()
    manager.run()
