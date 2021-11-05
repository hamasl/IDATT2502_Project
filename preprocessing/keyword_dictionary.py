import os
from yaml import safe_load
from os.path import isfile


def get():
    keyword_file = os.path.join(os.path.dirname(__file__),"keywords.yaml")
    keyword_dict = []
    if isfile(keyword_file):
        with open(keyword_file, mode='r') as stream:
            out = safe_load(stream)
            for x in out['keywords']:
                keyword_dict.append(x)
            for x in out['vulnerable_functions']:
                keyword_dict.append(x)
    return keyword_dict
