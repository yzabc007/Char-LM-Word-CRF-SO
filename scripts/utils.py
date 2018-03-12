import tensorflow as tf
import numpy as np
import time
import subprocess as sp
from collections import Counter
from nltk.tokenize import *

from conlleval_py import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_feed_dict(model, Model_Parameters, sents_idx_data):
    feed_dict = {}
    #
    sents_batch = []
    tags_batch = []
    for sent_data in sents_idx_data:
        sents_batch.append(sent_data['word_ids'])
        tags_batch.append(sent_data['tag_ids'])

    sents_batch, seq_length = pad_sentence_words(sents_batch)
    tags_batch = pad_tags(tags_batch)
    feed_dict[model.word_input_ids] = sents_batch
    feed_dict[model.tag_input_ids] = tags_batch
    feed_dict[model.sequence_lengths] = seq_length
    feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']

    if Model_Parameters['char_encode'] != 'None' and not Model_Parameters['use_hier_char']:
        chars_batch = []
        for sent_data in sents_idx_data:
            chars_batch.append(sent_data['char_ids'])

        if Model_Parameters['char_encode'] == 'lstm':
            char_id_batch, word_lengths = pad_word_chars(chars_batch)
            feed_dict[model.char_input_ids] = char_id_batch
            feed_dict[model.word_lengths] = word_lengths
        elif Model_Parameters['char_encode'] == 'cnn':
            char_id_batch, word_lengths = pad_word_chars(chars_batch, max_char_len=Model_Parameters['max_char_len'])
            feed_dict[model.char_input_ids] = char_id_batch
            feed_dict[model.word_lengths] = word_lengths

    if Model_Parameters['use_hier_char']:
        chars_batch = []
        for sent_data in sents_idx_data:
            chars_batch.append(sent_data['char_ids'])

        char_id_batch, char_lengths, word_pos_for, word_pos_bak = pad_word_chars_hierarchy(chars_batch)
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.char_lengths] = char_lengths
        feed_dict[model.word_pos_for] = word_pos_for
        feed_dict[model.word_pos_bak] = word_pos_bak

        if Model_Parameters['char_lm']:
            char_lm_forward = []
            char_lm_backward = []
            for sent_data in sents_idx_data:
                char_lm_forward.append(sent_data['char_lm_forward'])
                char_lm_backward.append(sent_data['char_lm_backward'])

            batch_char_lm_for = pad_tags(char_lm_forward)
            batch_char_lm_bak = pad_tags(char_lm_backward)

            feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']
            feed_dict[model.char_lm_forward] = batch_char_lm_for
            feed_dict[model.char_lm_backward] = batch_char_lm_bak

    if Model_Parameters['word_lm']:
        for_sents_batch = []
        bak_sents_batch = []
        for sent_data in sents_idx_data:
            for_sents_batch.append(sent_data['forward_words'])
            bak_sents_batch.append(sent_data['backward_words'])

        feed_dict[model.forward_words] = pad_tags(for_sents_batch)
        feed_dict[model.backward_words] = pad_tags(bak_sents_batch)

    # elif Model_Parameters['use_char_alone_model']:
    #     chars_batch = []
    #     char_tags_batch = []    #     feed_dict[model.tag_input_ids] = char_tags_batch
    #
    #     # feed_dict[model.word_input_ids] = words_batch
    #     # feed_dict[model.space_pos] = space_positions
    #     # feed_dict[model.word_tag_input_ids] = tags_batch

    #     tags_batch = []
    #     words_batch = []
    #
    #     for sent_data in sents_idx_data:
    #         chars_batch.append(sent_data['char_ids'])
    #         tags_batch.append(sent_data['tag_ids'])
    #         char_tags_batch.append(sent_data['char_tag_ids'])
    #         # words_batch.append(sent_data['word_for_char'])
    #
    #     tags_batch = pad_tags(tags_batch)
    #     char_id_batch, char_lengths = pad_characters_alone(chars_batch)
    #     char_tags_batch, _ = pad_characters_alone(char_tags_batch, tag=True)
    #     # words_batch, _ = pad_characters_alone(words_batch)
    #
    #     feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']
    #     feed_dict[model.char_input_ids] = char_id_batch
    #     feed_dict[model.sequence_lengths] = char_lengths

    return feed_dict


def pad_sentence_words(sents, max_sent_len=None, type=None):
    if not max_sent_len:
        max_sent_len = max([len(sent) for sent in sents])
    batch_sents = []
    batch_seq_length = []
    for sent in sents:
        batch_seq_length.append(len(sent))
        if type == 'float':
            batch_sents.append(sent + [0.0] * (max_sent_len - len(sent)))
        else:
            batch_sents.append(sent + [0] * (max_sent_len - len(sent)))
    return batch_sents, batch_seq_length


def pad_tags(tags, max_tag_len=None):
    if not max_tag_len:
        max_tag_len = max([len(tag) for tag in tags])
    batch_tags = []
    for tag in tags:
        batch_tags.append(tag + [0] * (max_tag_len - len(tag)))
    return batch_tags


def pad_word_chars(chars_batch, max_sent_len=None, max_char_len=None):
    if not max_sent_len:
        max_sent_len = max([len(sent) for sent in chars_batch])
    if not max_char_len:
        max_char_len = max([len(w) for sent in chars_batch for w in sent])
    batch_chars = []
    batch_word_length = []
    # padd zero words [[1,2,3,4], [4,5,0,0], [0,0,0,0]] with [4, 2, 0]
    for sent in chars_batch:
        batch_word_length.append([len(w) for w in sent] + [0] * (max_sent_len - len(sent)))
        cur_sent_char = []
        for word in sent:
            cur_sent_char.append(word + [0] * (max_char_len - len(word)))
        if len(sent) < max_sent_len:
            cur_sent_char.extend([[0] * max_char_len] * (max_sent_len - len(sent)))
        batch_chars.append(cur_sent_char)

    return batch_chars, batch_word_length


def pad_word_chars_hierarchy(chars_batch):
    max_sent_len = max([len(sent) for sent in chars_batch])
    max_char_len = max([len(np.concatenate(sent)) for sent in chars_batch])

    char_id_batch = []
    char_lengths = []
    word_pos_for = []
    word_pos_bak = []
    for idx, sent in enumerate(chars_batch):
        chars = np.concatenate(sent)
        char_id_batch.append(np.concatenate((chars, [0]*(max_char_len-len(chars)))))
        char_lengths.append(len(chars))

        word_pos_vec = np.cumsum([len(w) for w in sent]) - 1
        word_pos_vec = np.concatenate((word_pos_vec, [0]*(max_sent_len - len(word_pos_vec))))
        word_pos_for.append([[idx, pos] for pos in word_pos_vec])

        word_pos_vec = np.cumsum([len(w) for w in sent])[:-1]
        word_pos_vec = np.concatenate(([0], word_pos_vec, [0]*(max_sent_len - len(word_pos_vec) - 1)))
        word_pos_bak.append([[idx, pos] for pos in word_pos_vec])

    return char_id_batch, char_lengths, word_pos_for, word_pos_bak


def pad_characters_alone(chars_batch, tag=False):
    max_char_len = max([len(np.concatenate(sent)) for sent in chars_batch])

    char_id_batch = []
    char_lengths = []
    space_positions = []
    for idx, sent in enumerate(chars_batch):
        chars = np.concatenate(sent)
        char_id_batch.append(np.concatenate((chars, [0]*(max_char_len-len(chars)))))
        if not tag:
            char_lengths.append(len(chars))
            # word_pos_vec = np.cumsum([len(w) for w in sent]) - 1
            # word_pos_vec = np.concatenate((word_pos_vec, [0]*(max_sent_len - len(word_pos_vec))))
            # space_positions.append(word_pos_vec)

    return char_id_batch, char_lengths


def get_max_sent_length(sentences):
    return max([len(sent) for sent in sentences])


def get_max_char_length(sentences):
    return max([max([len(w[0]) for w in sent]) for sent in sentences])


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def plotting_storing(Model_Parameters, loss_train_record, loss_val_record, loss_test_record,
                                        f1_train_record, f1_val_record, f1_test_record):
    num_points = len(loss_train_record)
    x_list = range(1, len(loss_train_record) + 1)
    colors_train = []
    colors_val = []
    colors_test = []
    interval_idx = 0
    for idx in x_list:
        interval_idx += 1
        if interval_idx - 1 == Model_Parameters['freq_eval']:
            colors_train.append('b')
            colors_val.append('m')
            colors_test.append('g')
            interval_idx = 0
        else:
            colors_train.append('b')
            colors_val.append('m')
            colors_test.append('g')

    plt.figure(figsize=(12, 5))
    figure_name = Model_Parameters['fig_path'] + 'loss_' + Model_Parameters['fig_name'] + '.png'
    # plt.scatter(x=[1,2,3,4], y=[3,4,5,5], c=[1,2,3,4], marker='+')
    plt.scatter(x=x_list, y=loss_train_record, c=colors_train, marker='o', label='Training loss')
    plt.scatter(x=x_list, y=loss_val_record, c=colors_val, marker='s', label='Validaton loss')
    plt.scatter(x=x_list, y=loss_test_record, c=colors_test, marker='v', label='Testing loss')
    plt.legend(loc=1)
    plt.title('Loss covergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(figure_name)
    # sp.call('cp ' + figure_name + ' $PBS_O_WORKDIR/figures', shell=True)

    plt.figure(figsize=(12, 5))
    figure_name = Model_Parameters['fig_path'] + 'f1_' + Model_Parameters['fig_name'] + '.png'
    # plt.scatter(x=[1,2,3,4], y=[3,4,5,5], c=[1,2,3,4], marker='+')
    plt.scatter(x=x_list, y=f1_train_record, c=colors_train, marker='o', label='Training F1')
    plt.scatter(x=x_list, y=f1_val_record, c=colors_val, marker='s', label='Validation F1')
    plt.scatter(x=x_list, y=f1_test_record, c=colors_test, marker='v', label='Testing F1')
    x_best = f1_val_record.index(max(f1_val_record)) + 1
    val_best = max(f1_val_record)
    train_best = f1_train_record[x_best - 1]
    test_best = f1_test_record[x_best - 1]
    # plt.annotate(str(train_best), xy=(x_best, train_best), xytext=(x_best-(len(x_list) / 5), train_best), arrowprops=dict(arrowstyle='->', color='b'))
    plt.annotate(str(max(f1_val_record)), xy=(x_best, val_best), xytext=(x_best-(len(x_list) / 5)-1, val_best-5), arrowprops=dict(arrowstyle='->', color='m'))
    # plt.annotate(str(test_best), xy=(x_best, test_best), xytext=(x_best-(len(x_list) / 5)-2, test_best), arrowprops=dict(arrowstyle='->', color='g'))
    plt.legend(loc=2)
    plt.title('F1 score convergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('F1')
    plt.savefig(figure_name)
    # sp.call('cp ' + figure_name + ' $PBS_O_WORKDIR/figures', shell=True)
    # print 'Finish stroing new figures!'
    return


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
