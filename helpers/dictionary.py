__author__ = 'suraj'

import operator



def sort_dic_by_value(dic, reverse=False):
    return sorted(dic.iteritems(), key=operator.itemgetter(1), reverse=reverse)


# Maximum value of a dictionary
def dict_max(dic):
    aux = dict(map(lambda item: (item[1], item[0]), dic.items()))
    if not aux.keys():
        return 0
    max_value = max(aux.keys())
    return max_value, aux[max_value]
