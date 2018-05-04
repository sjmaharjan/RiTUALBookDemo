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




def build_drive_link_mapping():
    map={}
    html_doc = "/Users/suraj/Desktop/del/Content_Single_File_Text - Google Drive.html"
    with open(html_doc, "r") as fhandle:
        soup = BeautifulSoup(fhandle)
    with open('/Users/suraj/Documents/workspace/colombia/small_scripts/book_id.txt') as f_in:
         for line in f_in:
             url=get_google_drive_link_from_line(soup,line)
             map[line.rsplit('_',1)[0].strip()]=url
    return map



#	ISBN_10	ISBN_13	Title	PublicationDate	Author	Description	Genre	PageCount	PrintType	CoverImage	URL	Publisher	AccessViewStatus

def get_google_drive_link_from_line(soup,line):
    line = line.strip()
    book_fname = line.split(".")[0]
    div = soup.find("div", {"aria-label": book_fname + " Shared Text"})
    if div:
        # return "\thttps://drive.google.com/uc?export=view&id=" + div["data-id"]
        # https://drive.google.com/a/booxby.com/file/d/0B5mNQoFgvKShMUxtaERtYUlrQVU/view?usp=sharing
        return "https://drive.google.com/a/booxby.com/file/d/" + div["data-id"] + "/view?usp=sharing"
    else:
        print (line + "  Not found")
        return ""


if __name__ == '__main__':
    print( download_meta_info('VgLKYawnwHgC'))