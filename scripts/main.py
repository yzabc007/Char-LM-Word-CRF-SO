import sys
import os
import numpy as np
import argparse
from collections import OrderedDict
import time
import cPickle

import tensorflow as tf
from bilstm_model import *
from bilstm_att_model import *
from utils import *
from loader import *
from conlleval_py import *
from train import *
from config import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    Model_Parameters = Config()
    np.random.seed(Model_Parameters['random_seed'])

    if Model_Parameters['train_file_path'][-3::] == 'pkl':
        train_sents = cPickle.load(open(Model_Parameters['train_file_path']))
        val_sents = cPickle.load(open(Model_Parameters['val_file_path']))
        test_sents = cPickle.load(open(Model_Parameters['test_file_path']))
    else:
        # Load sentences
        train_sents = load_conll2003(Model_Parameters['train_file_path'])
        val_sents = load_conll2003(Model_Parameters['val_file_path'])
        test_sents = load_conll2003(Model_Parameters['test_file_path'])

        # Use selected tagging scheme (IOB / IOBES)
        update_tag_scheme(train_sents, Model_Parameters['tag_scheme'])
        update_tag_scheme(val_sents, Model_Parameters['tag_scheme'])
        update_tag_scheme(test_sents, Model_Parameters['tag_scheme'])

    print 'Data loaded!'

    print 'Training size: ', len(train_sents)
    print 'Val size:', len(val_sents)
    print 'Test size: ', len(test_sents)
    print 'An exmaple: ', train_sents[0]

    if Model_Parameters['train_size'] and Model_Parameters['train_size'] < len(train_sents):
        train_sents = train_sents[:Model_Parameters['train_size']]
    if Model_Parameters['val_size'] and Model_Parameters['val_size'] < len(val_sents):
        val_sents = val_sents[:Model_Parameters['val_size']]
    if Model_Parameters['test_size'] and Model_Parameters['test_size'] < len(test_sents):
        test_sents = test_sents[:Model_Parameters['test_size']]

    # create mapping
    if Model_Parameters['pre_trained_path']:
        print 'Begin mapping from glove ...'
        dic_words, word_to_id, id_to_word, W = word_mapping_glove(train_sents, Model_Parameters)
        Model_Parameters['embedding_initializer'] = W.astype(np.float32)
    else:
        print 'Begin mapping from random ...'
        dic_words, word_to_id, id_to_word = word_mapping_random(train_sents)
    Model_Parameters['vocab_size'] = len(word_to_id)
    singletons = set([w[0] for w in dic_words.items() if w[1] == 1])
    print 'Word mapped!'

    dic_tags, tag_to_id, id_to_tag = tag_mapping(train_sents)
    Model_Parameters['tag_size'] = len(tag_to_id)
    Model_Parameters['id_to_word_tag'] = id_to_tag
    print 'Tag id: ', id_to_tag
    print 'Tag mapped!'

    dic_chars, char_to_id, id_to_char = char_mapping(train_sents)
    # print char_to_id
    Model_Parameters['char_vocab_size'] = len(char_to_id)
    print 'Character mapped!'

    print 'Mapping finished!'

    # index data
    train_idx_data = make_idx_data(Model_Parameters, train_sents, word_to_id,
                                   tag_to_id, char_to_id=char_to_id, singletons=singletons)
    val_idx_data = make_idx_data(Model_Parameters, val_sents, word_to_id,
                                 tag_to_id, char_to_id=char_to_id, singletons=singletons)
    test_idx_data = make_idx_data(Model_Parameters, test_sents, word_to_id,
                                  tag_to_id, char_to_id=char_to_id, singletons=singletons)
    print 'Finish digitizing!'
    print 'An example: ', train_idx_data[0]

    if Model_Parameters['char_attention']:
        print 'Using character attention'

    model = bilstm(Model_Parameters)

    print 'Begin training ...'
    train(model, Model_Parameters, train_idx_data, val_idx_data, test_idx_data)

    return


if __name__ == '__main__':
    main()
