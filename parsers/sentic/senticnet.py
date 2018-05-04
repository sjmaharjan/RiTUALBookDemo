# -*- coding: utf-8 -*-
import os
from string import Template
import rdflib
import datetime

__author__ = 'suraj'


class Sentics(object):
    def __init__(self):
        dirname= os.path.dirname(__file__)
        self.sentic_local = os.path.join(dirname,'resources', "senticnet3.rdf.xml")
        self.parsed_graph = rdflib.Graph().parse(self.sentic_local, format="xml")
        self.query_base = Template('PREFIX sentic: <http://sentic.net/api> ' \
                                   'SELECT ?pleasantness ?attention ?sensitivity ?aptitude ?polarity ' \
                                   'WHERE { ' \
                                   '?concept sentic:text "$concept"; ' \
                                   'sentic:pleasantness ?pleasantness; ' \
                                   'sentic:attention ?attention; ' \
                                   'sentic:sensitivity ?sensitivity; ' \
                                   'sentic:aptitude ?aptitude; ' \
                                   'sentic:polarity ?polarity. ' \
                                   '}')

    def lookup(self, concept):
        query_str = self.query_base.substitute(concept=concept)
        query = self.parsed_graph.query(str(query_str))
        if len(query) == 0: return None
        return dict((str(sentic), float(score)) for (sentic, score) in query._get_bindings()[0].iteritems())


if __name__ == '__main__':
    start_time=datetime.datetime.now()
    # print("START %s" % start_time)
    sentic = Sentics()
    end_time=datetime.datetime.now()
    print end_time-start_time
    concepts = ['love', 'cat','a_little']
    start_time=datetime.datetime.now()
    for i in range(500):
        for concept in concepts:
            sentic.lookup(concept)
    end_time=datetime.datetime.now()
    print end_time-start_time