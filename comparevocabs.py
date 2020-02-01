#!/usr/bin/env python3

import sys
import os
import re

from collections import OrderedDict
from itertools import combinations
from random import sample
from logging import warning

from berttokenizer import basic_tokenize


# BERT special tokens
BERT_SPECIAL = set(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])

# BERT "unused" special token format
UNUSED_RE = re.compile(r'^\[unused[0-9]+\]$')


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('vocab', nargs='+', help='BERT vocabulary')
    return ap


def check_vocab(vocab, name):
    multipiece = []
    for v in vocab:
        t = v if not v.startswith('##') else v[2:]
        if len(basic_tokenize(t)) > 1:
            multipiece.append(v)
    if multipiece:
        warning('{} contains {} items containing multiple basic tokens: {}'.\
                format(name, len(multipiece), ' '.join(multipiece)))


def load_vocab(path):
    vocab = []
    with open(path) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if not l or l.isspace():
                warning('skipping empty line {} in {}: "{}"'.format(ln, path, l))
                continue
            if not l or any(c.isspace() for c in l):
                warning('line {} in {}: "{}"'.format(ln, path, l))
            vocab.append(l)
    print('read {} from {}'.format(len(vocab), path), file=sys.stderr)
    return vocab


def filter_special(vocab, name):
    filtered, special, unused = [], [], []
    for v in vocab:
        if v in BERT_SPECIAL:
            special.append(v)
        elif UNUSED_RE.match(v):
            unused.append(v)
        else:
            filtered.append(v)
    print('{}: {} special, {} unused, {} other'.format(
        name, len(special), len(unused), len(filtered)), file=sys.stderr)
    return filtered


def max_sample(population, k):
    return sample(population, min(len(population), k))


def compare(vocab1, vocab2, name1, name2):
    v1 = set(vocab1)
    v2 = set(vocab2)
    print('{} ({}) / {} ({}): overlap {} ({:.1%}/{:.1%})'.format(
        name1, len(v1), name2, len(v2), len(v1&v2), len(v1&v2)/len(v1),
        len(v1&v2)/len(v2)))
    print('{} \ {}: {}'.format(name1, name2, ' '.join(max_sample(v1-v2, 10))))
    print('{} \ {}: {}'.format(name2, name1, ' '.join(max_sample(v2-v1, 10))))


def main(argv):
    args = argparser().parse_args(argv[1:])
    if len(args.vocab) < 2:
        print('require at least 2 vocabs to compare', file=sys.stderr)
        return 1
    vocabs = OrderedDict([(p, load_vocab(p)) for p in args.vocab])
    for k in vocabs:
        vocabs[k] = filter_special(vocabs[k], k)
    for k in vocabs:
        check_vocab(vocabs[k], k)
    for k1, k2 in combinations(vocabs.keys(), 2):
        compare(vocabs[k1], vocabs[k2], k1, k2)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
