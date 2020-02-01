#!/usr/bin/env python3

import sys
import os
import re

from collections import Counter, OrderedDict
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
    ap.add_argument('-t', '--reference-text', default=None,
                    help='Reference text to compare against')
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


def compare(vocab1, vocab2, name1, name2, token_counts):
    v1 = set(vocab1)
    v2 = set(vocab2)
    print('{} ({}) / {} ({}): overlap {} ({:.1%}/{:.1%})'.format(
        name1, len(v1), name2, len(v2), len(v1&v2), len(v1&v2)/len(v1),
        len(v1&v2)/len(v2)))
    max_examples = 10
    if token_counts is None:
        # No reference text, output a sample of differences
        print('{} \ {}: {}'.format(name1, name2, ' '.join(
            max_sample(v1-v2, max_examples))))
        print('{} \ {}: {}'.format(name2, name1, ' '.join(
            max_sample(v2-v1, max_examples))))
    else:
        # Print differences ordered by token frequency
        v1count, v2count = 0, 0
        for i, (t, _) in enumerate(sorted(token_counts.items(),
                                          key=lambda i:-i[1]),
                                   start=1):
            if t in v1 and t not in v2 and v1count < max_examples:
                print('{} \ {}: {} (rank {})'.format(name1, name2, t, i))
                v1count += 1
            if t in v2 and t not in v1 and v2count < max_examples:
                print('{} \ {}: {} (rank {})'.format(name2, name1, t, i))
                v2count += 1
            if v1count >= max_examples and v2count >=max_examples:
                break


def basictoken_counts(fn):
    counts = Counter()
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip()
            for t in basic_tokenize(l):
                if t and not t.isspace():
                    counts[t] += 1
    return counts


def main(argv):
    args = argparser().parse_args(argv[1:])
    if len(args.vocab) < 2:
        print('require at least 2 vocabs to compare', file=sys.stderr)
        return 1
    vocabs = OrderedDict([(p, load_vocab(p)) for p in args.vocab])
    if args.reference_text is None:
        reference_counts = None
    else:
        reference_counts = basictoken_counts(args.reference_text)
    for k in vocabs:
        vocabs[k] = filter_special(vocabs[k], k)
    for k in vocabs:
        check_vocab(vocabs[k], k)
    for k1, k2 in combinations(vocabs.keys(), 2):
        compare(vocabs[k1], vocabs[k2], k1, k2, reference_counts)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
