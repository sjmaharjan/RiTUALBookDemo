# -*- coding: utf-8 -*-
import codecs
import urllib2
import os
import networkx as nx
import nltk

__author__ = 'suraj'

class Concept(object):
    """
    Responsibility: to download/load concepts from sentic api
                    save/load concepts
    """
    def __init__(self):
        dirname= os.path.dirname(__file__)
        self.url = 'http://sentic.net/api/en/concept'
        self.concept_file=os.path.join(dirname,'resources','Concepts.txt')
        self.concept_rdf=os.path.join(dirname,'resources','concepts.rdf.xml')

    def __download_concepts(self):
        html=[]
        if os.path.isfile(self.concept_rdf):
            with codecs.open(self.concept_rdf,encoding='utf-8',mode='r') as f:
                html=f.readlines()
        else:
            response = urllib2.urlopen(self.url)
            html = response.readlines()
        concepts_lst = []
        for line in html:
            # print (line)
            line=line.strip('\r\n')
            concept=line.rsplit('/',1)[-1]
            concept=concept.replace('">','')
            concepts_lst.append(concept.replace("_", " "))

        del concepts_lst[0:2]
        del concepts_lst[-1]

        print ("Number of Concepts: ", len(concepts_lst))
        return concepts_lst

    def save_concepts(self):
        with codecs.open(self.concept_file,encoding='utf-8',mode='w') as f:
            for concept  in self.__download_concepts():
               f.write("%s\n"%concept)



    def get_concepts(self):
        concepts=[]
        if os.path.isfile(self.concept_file):
            with codecs.open(self.concept_file,encoding='utf-8',mode='r') as f:
                for line in f:
                    concepts.append(line.strip('\n'))

        else:
            concepts=self.__download_concepts()

        return concepts






class ConceptGraph(object):
    def __init__(self):
        dirname= os.path.dirname(__file__)
        self.concepts = Concept().get_concepts()
        self.concept_graph_dump=os.path.join(dirname,'resources','concept_graph.gpkl')

    def create_concept_graph(self, concepts=None):
        if concepts:
            concepts_to_graph = concepts
        else:
            concepts_to_graph = self.concepts

        NodeGraph = nx.DiGraph()

        for word in concepts_to_graph:
            nodes = nltk.word_tokenize(word)
            if (NodeGraph.has_edge("root", nodes[0])):
                for i in range(len(nodes) - 1):
                    NodeGraph.add_edge(nodes[i], nodes[i + 1])

            else:
                NodeGraph.add_edge("root", nodes[0])
                for i in range(len(nodes) - 1):
                    NodeGraph.add_edge(nodes[i], nodes[i + 1])

        return NodeGraph

    def save(self,concepts=None):
        graph=self.create_concept_graph(concepts)
        nx.write_gpickle(graph,self.concept_graph_dump)


    def load(self):
        G = nx.read_gpickle( self.concept_graph_dump)
        return G

    @staticmethod
    def check_in_graph( Graph, concepts):
        if (concepts):
            ToCheck = concepts
        else:
            print ("Sorry ! The knowledge base does not contain any concept for your query")

        ConceptsToBeSearched = []

        for i in range(len(ToCheck)):
            tokens = nltk.word_tokenize(ToCheck[i])
            length = len(tokens)
            print length
            if (length > 1):
                for j in range(len(tokens) - 1):
                    if (Graph.has_edge(tokens[j], tokens[j + 1])):
                        print "Match Found in Database"
                        s = tokens[j] + " " + tokens[j + 1]
                        ConceptsToBeSearched.append(s)

        return ConceptsToBeSearched




if __name__ == '__main__':
    # concepts=Concept()
    # # concepts.save_concepts()
    # print concepts.get_concepts()
    CG=ConceptGraph()
    CG.save()
    G=CG.load()
    # nx.draw(G)
    print (ConceptGraph.check_in_graph(G,['a lot','love']))
