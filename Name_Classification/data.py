import torch
import glob
import unicodedata
import string
import os
import torch as T

## All possible characters which could be used are set as global variables
all_letters = string.ascii_letters + ".,;"
n_letters = len(all_letters)

def find_files(path):
    return glob.glob(path)

def unicode_to_ascii(s):
    """ Removes all accents and dialectics
        https://stackoverflow.com/a/518232/2809427
    """
    return "".join( c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn"
                    and c in all_letters
                  )

def read_lines(filenm):
    lines = open(filenm, encoding="utf-8").read().strip().split("\n")
    return [ unicode_to_ascii(line) for line in lines ]

def letter_to_idx(letter):
    return all_letters.find(letter)

def word_to_tensor(word):
    """ A word is a sequence, so it has dimensions [sq_len, batch=1, n_letters]
    """
    word_tens = T.zeros( len(word), 1, n_letters )
    for ln, letter in enumerate(word):
        word_tens[ln][0][letter_to_idx(letter)] = 1
    return word_tens

def build_dataset(all_files):
    class_dict = {}
    language_list = []

    for filename in find_files(all_files):

        language = os.path.splitext(os.path.basename(filename))[0]
        names = read_lines( filename )

        language_list.append(language)
        class_dict[language] = names

    return language_list, class_dict
