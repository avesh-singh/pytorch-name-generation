from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import random


def find_files(path): return glob.glob(path)


all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


category_lines = {}
all_categories = []


def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_category = len(all_categories)


def word_to_tensor(name):
    idxs = [all_letters.find(a) for a in name]
    oh_tensor = torch.zeros(len(idxs), 1, n_letters, dtype=torch.float)
    for i, id in enumerate(idxs):
        oh_tensor[i, 0, id] = 1
    return oh_tensor


def category_to_tensor(category):
    idx = all_categories.index(category)
    oh_tensor = torch.zeros(1, n_category, dtype=torch.float)
    oh_tensor[0, idx] = 1
    return oh_tensor


def get_target_tensor(name):
    idxs = [all_letters.find(a) for a in name]
    idxs.append(n_letters - 1)
    return torch.tensor(idxs[1:], dtype=torch.long)


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example():
    category = random_choice(all_categories)
    name = random_choice(category_lines[category])
    category_tensor = category_to_tensor(category)
    target_tensor = get_target_tensor(name)
    name_tensor = word_to_tensor(name)
    return category_tensor, name_tensor, target_tensor, category, name
