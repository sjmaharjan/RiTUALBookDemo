# -*- coding: utf-8 -*-
from __future__ import division, print_function
from itertools import groupby
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import enchant
from collections import defaultdict
import os
import codecs
import io
import json
import math
import networkx as nx
from concepts import ConceptGraph
from senticnet import Sentics
import logging
from nltk.corpus import conll2000
import multiprocessing
from nltk.chunk.util import conlltags2tree
from manage import app



__author__ = 'suraj'
__all__ = ['run_sentic_parser', 'sentic_parse', 'ChunkParser', 'VerbChunk', 'SenticsParser']

logger = logging.getLogger("sentic-parser")
logger.setLevel(logging.INFO)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)


def bigram_concepts(concepts, tags):
    # if tags:
    # sentence = zip(concepts.split(), tags.split())
    # else:
    # sentence = tagger.tag(nltk.word_tokenize(concepts))

    # print "bigram",sentence
    sentence = zip(concepts.split(), tags.split())
    Bigrams = []

    for i in range(len(sentence) - 1):
        if (sentence[i][1] == "JJ" and sentence[i + 1][0] in stopwords.words(
                'english')):  # If the bigram is [ adj + stopword ] , ignore
            continue  # bigrams like "a very" are ignored

        elif (sentence[i][0] in stopwords.words('english') and sentence[i + 1][0] in stopwords.words(
                'english')):  # If the bigram is [ adj + stopword ] , ignore
            continue


        elif (sentence[i + 1][1] == "JJ" and sentence[i][0] in stopwords.words(
                'english')):  # If the bigram is [ stopword + adj ] , ignore
            continue  # bigrams like "amazingly a" is ignored

        elif (sentence[i][1] == "JJ" and sentence[i + 1][1].startswith(
                "NN")):  # If the bigram is [ adj + concept ] , then include [adj + concept] and [concept] to the list
            Bigrams.append(sentence[i + 1][
                               0])  # e.g) "special christmas" --> concepts extracted will be "special christmas" and "christmas" are added
            Bigrams.append(sentence[i][0] + " " + sentence[i + 1][0])

        elif (sentence[i][0] in stopwords.words("english") and sentence[i + 1][1].startswith(
                "NN")):  # If the bigram is [ stopword + concept ], then inlcude only the concept w/ and w/o the concept
            Bigrams.append(sentence[i + 1][
                               0])  # e.g) "the christmas" --> concepts that will be extracted is "christmas" , "the christmas"
            Bigrams.append(sentence[i][0] + " " + sentence[i + 1][0])

        elif (sentence[i][1].startswith("NN") and sentence[i + 1][
            1] == "JJ"):  # If the bigram ends with adjective , then ignore the adjective.
            Bigrams.append(sentence[i][0])  # e.g) "present amazing" --> concept that will be extracted is "present"

        elif (sentence[i][1].startswith("NN") and sentence[i + 1][0] in stopwords.words(
                "english")):  # If the bigram ends with a stopword , then ignore the stopword
            Bigrams.append(sentence[i][0])  # e.g) "christmas the" --> concept that will be extracted is "christmas"

        else:
            Bigrams.append(sentence[i][0] + " " + sentence[i + 1][0])

    # print Bigrams

    return Bigrams


def verb_stem(word):
    st = LancasterStemmer()
    StemmedVerb = st.stem(word)
    dic = enchant.Dict("en_US")
    if (dic.check(StemmedVerb)):
        return StemmedVerb
    else:
        return StemmedVerb + "e"


def find_split(sentence, TaggedSentence):
    TokenizedSentence = nltk.word_tokenize(sentence)

    SplitList = []
    SentAdded = ""
    split = 0

    # print TaggedSentence

    for i in range(len(TaggedSentence)):
        if TaggedSentence[i][1].startswith("VB"):
            SplitList.append(SentAdded)
            try:
                if (TaggedSentence[i + 1][1].startswith("VB")):
                    SentAdded = ""
                else:
                    SplitList.append(SentAdded)
                    SentAdded = TaggedSentence[i][0] + '/' + TaggedSentence[i][1] + " "
                    #	print "split"
            except:
                SplitList.append(TaggedSentence[i][0] + '/' + TaggedSentence[i][1])

        else:
            # print SentAdded
            SentAdded = SentAdded + TokenizedSentence[i] + '/' + TaggedSentence[i][1] + " "

    SplitList.append(SentAdded)

    Str_list = filter(None, SplitList)
    Str_list = list(set(Str_list))

    '''
    for i in range(len(Str_list)):
        Str_list[i] = Str_list[i][:-1].translate(string.maketrans("",""), string.punctuation)
    '''
    return Str_list


class ChunkParser(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(sentence, chunktags)]
        return conlltags2tree(conlltags)


class VerbChunk():
    def __init(self):
        pass

    def VerbParse(self, TaggedChunk):
        grammar = r"""
		  		VP: {<VB.*>+ } # Chunk verbs
				"""
        cp = nltk.RegexpParser(grammar)  # ,loop=2)
        vpTree = cp.parse(TaggedChunk)
        return vpTree


def getOutputConcepts(sentence, tags, np_chunker, vp_chunker):
    # sentence = raw_input("Enter your search sentence ===>> " ).lower()
    # print sentence
    # Parse = SenticParser()
    # TaggedSentence = tagger.tag(sentence.split())
    TaggedSentence = zip(sentence.split(), tags.split())

    ConceptChunks = []
    NPChunker = np_chunker

    ChunkedSentences = find_split(sentence, TaggedSentence)

    events = []
    objects = []

    verb = ""

    # print ChunkedSentences

    for sent in ChunkedSentences:

        TaggedChunk = [(w_t.rsplit('/', 1)[0], w_t.rsplit('/', 1)[1]) for w_t in sent.split()]
        # print TaggedChunk

        PartsSentence = [l for l in [list(group) for key, group in groupby(TaggedChunk, key=lambda k: k[1] == "IN")]
                         if l[0][1] != 0]

        # print PartsSentence

        SplitByIN = []

        for j in (range(0, len(PartsSentence), 2)):
            SplitByIN.append(PartsSentence[j])

        # print "Split",SplitByIN


        NounTree = NPChunker.parse(SplitByIN[0])

        nouns = []

        for n in NounTree:
            if isinstance(n, nltk.tree.Tree):
                if n.label() == 'NP':
                    TaggedPhrase = n.leaves()
                    # print 'tagphrase',TaggedPhrase
                    TagRemoved = " ".join(tup[0] for tup in TaggedPhrase)
                    Tags = " ".join(tup[1] for tup in TaggedPhrase)
                    nouns.append(TagRemoved)
                    nouns = nouns + bigram_concepts(TagRemoved, Tags)
                    # print 'tag1',TagRemoved
            else:
                continue

        nouns = list(set(nouns) - set(["i", "you", "we", "our", "they", "their", "he", "she", "it", "her", "his"]))

        objects = objects + nouns

        for j in range(1, len(SplitByIN)):

            NounTree = NPChunker.parse(SplitByIN[j])

            other_nouns = []

            for n in NounTree:
                if isinstance(n, nltk.tree.Tree):
                    if n.label() == 'NP':
                        TaggedPhrase = n.leaves()
                        TagRemoved = " ".join(tup[0] for tup in TaggedPhrase)
                        Tags = " ".join(tup[1] for tup in TaggedPhrase)
                        other_nouns.append(TagRemoved)
                        other_nouns = other_nouns + bigram_concepts(TagRemoved, Tags)
                        # print 'tag2',TagRemoved
                else:
                    continue

            objects = objects + other_nouns

        # for i in NounTree:
        #	print n.leaves()



        vp = vp_chunker

        FindVerbTree = vp.VerbParse(TaggedChunk)

        # print FindVerbTree

        for n in FindVerbTree:
            if isinstance(n, nltk.tree.Tree):
                if n.label() == 'VP':
                    # print n.label()
                    verb = verb_stem(n.leaves()[-1][0])
                    # print verb
                else:
                    continue

        if (verb not in stopwords.words('english')):
            if (nouns):
                # print nouns
                # print verb
                for noun in nouns:
                    events.append(verb + " " + noun)

            else:
                events.append(verb)

    events = list(set(events))
    objects = list(set(objects))

    # print events
    # print objects

    outputList = list(set(events) | set(objects))
    # print outputList

    return outputList


##########################################################################################



class SenticsParser(object):
    def __init__(self):
        self.G = ConceptGraph().load()
        self.sn = Sentics()

    def get_sentics_of_sentence(self, sentence, tags, np_chunker, vp):

        words = sentence.split()

        list_concepts = []
        conc = []

        to_add = ""

        for word in words:
            if (word in self.G):
                conc.append(word)
                to_add += word + " "
            elif (to_add != ""):
                list_concepts.append(to_add[:-1])
                to_add = ""

        if (to_add != ""):
            list_concepts.append(to_add[:-1])

        parserList = getOutputConcepts(sentence, tags, np_chunker, vp)

        list_concept = list(set(list_concepts) | set(parserList))

        list_concept = filter(bool, list_concept)

        list_concept = set(list(list_concepts))

        to_search = []

        for phrase in list_concepts:
            concepts = phrase.split()
            to_search = to_search + concepts
            for i in range(len(concepts) - 1):
                for j in range(i + 1, len(concepts)):
                    try:
                        k = nx.dijkstra_path(self.G, concepts[i], concepts[j])
                        if (len(k) == j - i + 1 and k == concepts[i:j + 1]):
                            to_search = list(set(to_search) - set(k))
                            word_to_add = "_".join(k)
                            to_search.append(word_to_add)
                    except:
                        continue

        to_search = list(set(to_search))

        sorted_by_length = sorted(to_search, key=lambda tup: len(tup.split("_")))
        return sorted_by_length, filter(lambda x: x is not None, [self.sn.lookup(concept) for concept in to_search])


def extract(file_name):
    def rotate(line):
        if line == 'first':
            return 'second'
        if line == 'second':
            return 'third'
        if line == 'third':
            return 'first'

    word_pos = ''
    with codecs.open(file_name, mode='r', encoding='utf-8') as f_in:
        # print file_name
        new_line = 'first'
        buffer = ''
        for line in f_in.readlines():
            if not line.strip():
                if new_line == 'first':
                    word_pos += buffer
                if new_line == 'second':
                    pass
                    # parse_tree.append(buffer)
                if new_line == "third":
                    pass
                    # dependency.append(buffer)
                buffer = ''
                new_line = rotate(new_line)
            else:
                buffer += line
    return word_pos


def sentic_parse(filename, output_dir, sp, np, vp):
    dir_name, base_name = os.path.split(filename)
    fname = os.path.basename(filename) + '.json'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if os.path.exists(os.path.join(output_dir, fname)):
        logger.info('Already parsed %s' % fname)
        return fname
    else:

        word_pos = extract(filename)
        data = defaultdict(list)
        # print word_pos
        logger.info('start %s' % filename)
        for sentence_tags in word_pos.split("\n"):
            if not sentence_tags.startswith('(())'):
                try:
                    sentence = " ".join([sent.rsplit('/', 1)[0] for sent in sentence_tags.split()])
                    tags = " ".join([sent.rsplit('/', 1)[1] for sent in sentence_tags.split()])
                    concepts, scores = (sp.get_sentics_of_sentence(sentence.lower(), tags, np, vp))
                    if scores:
                        sentics_avg = []
                        # print scores
                        for key in ['sensitivity', 'attention', 'pleasantness', 'aptitude', 'polarity']:
                            sentics_avg.append(round(sum([s_dict[key] for s_dict in scores]) / float(len(scores)), 3))
                        data[os.path.basename(filename)].append({'c': concepts, 's': sentics_avg})

                    else:
                        data[os.path.basename(filename)].append({'c': concepts, 's': scores})
                except Exception as e:
                    print(e)
                    logger.error("Error for line %s :%s" % (sentence_tags, filename))
        print('End')

        with io.open(os.path.join(output_dir, fname), 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False))
        logger.info('done %s' % os.path.basename(filename))
        return os.path.basename(filename)


def read_parsed_files(filename):
    parsed_files = []
    with open(filename, mode='r') as f:
        for line in f:
            parsed_files.append(line.strip('\r\n'))
    return parsed_files


def produce_task(queue, dir_path, output_dir):
    def add_files(done_lst):
        for file in os.listdir(dir_path):
            if file.replace('_st_parser.txt', '_st_parser.txt.json') in done_lst:
                continue
            queue.put(os.path.join(dir_path, file))
            logger.info("producer [%s] putting value [%s] in queue..." % (multiprocessing.current_process().name, file))

    files_done = os.listdir(output_dir)
    add_files(files_done)


def consumer_task(queue, func, output_dir, sp, np, vp):
    while not queue.empty():
        work_args = queue.get(True, 0.1)
        logger.info(
            "consumer [%s] getting value [%s] from queue..." % (multiprocessing.current_process().name, work_args))
        func(work_args, output_dir, sp, np, vp)


def check_concept_dump():
    dirname = os.path.dirname(__file__)
    if os.path.exists(os.path.join(dirname, 'resources', 'concept_graph.gpkl')):
        return True
    else:
        return False


def run_sentic_parser(data_dir=None):
    if not data_dir:
        data_dir = app.config['STANFORD_PARSER_OUTPUT']
    output_dir = app.config['SENTIC_PARSE_OUTPUT']

    if not check_concept_dump():
        logger.info("Dumping concepts")
        ConceptGraph().save()

    manager = multiprocessing.Manager()
    data_queue = manager.Queue()

    producer = multiprocessing.Process(target=produce_task, args=(data_queue, data_dir, output_dir))
    producer.start()
    producer.join()

    logger.info("Number of jobs %s", data_queue.qsize())

    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

    # training the chunker
    NPChunker = ChunkParser(train_sents)
    vp = VerbChunk()

    sp = SenticsParser()

    # consumers
    consumer_list = []
    for i in range(multiprocessing.cpu_count()):
        consumer = multiprocessing.Process(target=consumer_task, args=(data_queue,
                                                                       sentic_parse, output_dir, sp, NPChunker, vp))
        consumer_list.append(consumer)
        consumer.start()

    [consumer.join() for consumer in consumer_list]

    logger.info("Done ...")
