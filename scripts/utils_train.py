import time
import cPickle

import tensorflow as tf
import numpy as np
import subprocess as sp

from utils import *
from loader import *
from conlleval_py import *


def predict(model, sess, feed_dict_):
    pred_tags_batch = []
    feed_dict_[model.dropout_keep_prob] = 1.0
    if model.params['use_crf_loss']:
        total_loss, logits, transition_params = sess.run([model.total_loss, model.logits, model.transition_params], feed_dict=feed_dict_)
        seq_lengths = feed_dict_[model.sequence_lengths]
        for idx_batch in xrange(len(seq_lengths)):
            seq_score = logits[idx_batch,:seq_lengths[idx_batch],:]
            # print seq_score.shape
            # print seq_score[0]
            pred_tags, pred_tags_score = tf.contrib.crf.viterbi_decode(seq_score, transition_params)
            pred_tags_batch.append(pred_tags)
    else:
        total_loss, pred_tags = sess.run([model.total_loss, model.pred_tags], feed_dict=feed_dict_)
        seq_lengths = feed_dict_[model.sequence_lengths]
        for idx_batch in xrange(len(seq_lengths)):
            pred_tags_batch.append(pred_tags[idx_batch, :seq_lengths[idx_batch]])

    return pred_tags_batch, [total_loss]


def predict_char(model, sess, feed_dict_):
    pred_tags_batch = []
    feed_dict_[model.dropout_keep_prob] = 1.0
    char_loss, logits, transition_params = sess.run([model.char_loss, model.char_logits, model.char_trainsition_params], feed_dict=feed_dict_)

    seq_lengths = feed_dict_[model.sequence_lengths]
    batch_size = len(seq_lengths)
    for idx_batch in xrange(batch_size):
        seq_score = logits[idx_batch,:seq_lengths[idx_batch], :]
        # print seq_score.shape
        # print seq_score[0]
        pred_tags, pred_tags_score = tf.contrib.crf.viterbi_decode(seq_score, transition_params)
        pred_tags_batch.append(pred_tags)

    return pred_tags_batch, [char_loss]


def predict_hier(model, sess, feed_dict_):
    pred_char_tags_batch = []
    pred_word_tags_batch = []
    feed_dict_[model.dropout_keep_prob] = 1.0
    char_loss, word_loss, char_logtis, word_logits, char_trainsition_params, word_transition_params = \
        sess.run([model.char_loss, model.word_loss, model.char_logits, model.word_logits,
            model.char_trainsition_params, model.word_transition_params], feed_dict=feed_dict_)

    word_lengths = feed_dict_[model.sequence_lengths]
    char_lengths = feed_dict_[model.char_lengths]
    batch_size = len(word_lengths)

    for idx_batch in xrange(batch_size):
        # character level
        char_seq_score = char_logtis[idx_batch, :char_lengths[idx_batch], :]
        pred_char_tags, _ = tf.contrib.crf.viterbi_decode(char_seq_score, char_trainsition_params)
        pred_char_tags_batch.append(pred_char_tags)

        # word level
        word_seq_score = word_logits[idx_batch, :word_lengths[idx_batch], :]
        word_pred_tags, _ = tf.contrib.crf.viterbi_decode(word_seq_score, word_transition_params)
        pred_word_tags_batch.append(word_pred_tags)

    return pred_char_tags_batch, pred_word_tags_batch, char_loss, word_loss


def predict_hier2(model, sess, feed_dict_):
    pred_char_tags_batch = []
    pred_word_tags_batch = []
    feed_dict_[model.dropout_keep_prob] = 1.0
    word_loss, word_logits, word_transition_params = \
        sess.run([model.word_loss, model.word_logits, model.word_transition_params], feed_dict=feed_dict_)

    word_lengths = feed_dict_[model.sequence_lengths]
    char_lengths = feed_dict_[model.char_lengths]
    batch_size = len(word_lengths)

    char_loss = 0

    for idx_batch in xrange(batch_size):
        # character level
        pred_char_tags = [0] * char_lengths[idx_batch]
        pred_char_tags_batch.append(pred_char_tags)

        # word level
        word_seq_score = word_logits[idx_batch, :word_lengths[idx_batch], :]
        word_pred_tags, _ = tf.contrib.crf.viterbi_decode(word_seq_score, word_transition_params)
        pred_word_tags_batch.append(word_pred_tags)

    return pred_char_tags_batch, pred_word_tags_batch, char_loss, word_loss
