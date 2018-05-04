__author__ = 'suraj'

from nltk.corpus import conll2000
import os
from bookweb import celery
from flask import current_app
from stanford_parser import standford_parser
from sentic.sentic_parser import sentic_parse, ChunkParser, VerbChunk, SenticsParser

__all__=['run_sentic_parser','run_stanford_parser']



@celery.task
def run_stanford_parser(fname):
    app=current_app._get_current_object()

    out_file=standford_parser(fname,app.config['STANFORD_PARSER_OUTPUT'])
    if out_file:
        return os.path.join(app.config['STANFORD_PARSER_OUTPUT'],out_file)
    return out_file




@celery.task
def run_sentic_parser(fname):
    app=current_app._get_current_object()
    if fname:

        test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
        train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

        NPChunker = ChunkParser(train_sents)
        vp = VerbChunk()

        sp = SenticsParser()

        out_fname=sentic_parse(fname,app.config['SENTIC_PARSE_OUTPUT'],sp, NPChunker, vp)
        if out_fname:
            return os.path.join(app.config['SENTIC_PARSE_OUTPUT'],out_fname)
        return out_fname
    else:
        return None
