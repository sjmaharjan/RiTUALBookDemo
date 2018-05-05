__author__ = 'suraj'

import requests
import json
import codecs
import os
import pandas as pd
from collections import OrderedDict
from bs4 import BeautifulSoup


__all__ =['download_meta_info']

GOOGLE_BOOK_URL = 'https://www.googleapis.com/books/v1/volumes/{id}'.format


def download_meta_info(book_id):
    url = GOOGLE_BOOK_URL(id=book_id)
    r = requests.get(url)
    print ("requesting", r.url)

    if r.status_code == requests.codes.ok:
        return r.json()
    else:
        r.raise_for_status()



