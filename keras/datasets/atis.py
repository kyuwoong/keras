from __future__ import absolute_import
from six.moves import cPickle
import gzip
from .data_utils import get_file
from six.moves import zip
import numpy as np


def load_data(path="atis.pkl.gz", nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):

    path = get_file(path, origin="http://www-etud.iro.umontreal.ca/~mesnilgr/atis/atis.pkl.gz")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set, test_set, dicts = cPickle.load(f)
    f.close()

    train_X = train_set[0]
    test_X = test_set[0]
    train_X += test_X

    w2idx = dicts['words2idx']

    idx2w  = dict((v,k) for k,v in w2idx.iteritems())

    train_sentences = []

    vocab = set()
    for sentence in train_X:
        if maxlen and len(sentence) > (maxlen-1):
            continue
        words = []
        for idx in sentence:
            words.append(idx2w[idx])

        # adding end of sentence marker
        words.append('<EOS>')
        train_sentences.append(words)
        vocab |= set(words)

    return train_sentences, sorted(vocab)
