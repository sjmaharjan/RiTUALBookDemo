# -*- coding: utf-8 -*-
import codecs
import json

__author__ = 'suraj'

def read_content(filename,encoding='utf-8'):
    content = ''
    with codecs.open(filename, mode='r', encoding=encoding) as f_in:
        content=f_in.read()
    return content

def extract(file_name):
    def rotate(line):
        if line == 'first':
            return 'second'
        if line == 'second':
            return 'third'
        if line == 'third':
            return 'first'

    word_pos, parse_tree, dependency = '', [], []
    with codecs.open(file_name, mode='rb', encoding='utf-8') as f_in:
        print(file_name)
        new_line = 'first'
        buffer = ''
        for line in f_in.readlines():

            if not line.strip():
                if new_line == 'first':
                    word_pos += buffer
                if new_line == 'second':
                    parse_tree.append(buffer)
                if new_line == "third":
                    dependency.append(buffer)
                buffer = ''
                new_line = rotate(new_line)
            else:
                buffer += line
    return word_pos, parse_tree, dependency


def pos_data(word_pos):
    pos_tags = ""
    for line in word_pos.splitlines():
        # print line
        for word_tag in line.strip().split():
            try:
                word, tag = word_tag.rsplit('/', 1)
            except ValueError as err:
                word, tag = word_tag, 'CD'
            # print word, tag
            # self.words += word + " "
            pos_tags += tag + " "
        pos_tags += '\n'
    return pos_tags




def load_concepts(filename):
    print ("Loading file in ",filename)
    c, sensitivity, attention, pleasantness, aptitude, polarity = '', [], [], [], [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # "Adventure_Stories_8493_st_parser.txt"

        for fname, concepts in data.iteritems():
            for concept in concepts:
                if concept['c']:
                    c += " ".join(concept['c']) + " "
                if concept['s']:
                    sensitivity.append(concept['s'][0])
                    attention.append(concept['s'][1])
                    pleasantness.append(concept['s'][2])
                    aptitude.append(concept['s'][3])
                    polarity.append(concept['s'][4])

    return c, sensitivity, attention, pleasantness, aptitude, polarity