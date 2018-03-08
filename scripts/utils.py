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
    # batch_size = len(sents_idx_data)
    # max_sent_len = max([len(sent_data['word_ids']) for sent_data in sents_idx_data])
    feed_dict = {}
    if Model_Parameters['use_hierarchy_lstm']:
        sents_batch = []
        tags_batch = []
        chars_batch = []
        char_tags_batch = []

        for_sents_batch = []
        bak_sents_batch = []

        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])
            chars_batch.append(sent_data['char_ids'])
            char_tags_batch.append(sent_data['char_tag_ids'])
            # for_sents_batch.append(sent_data['forward_words'])
            # bak_sents_batch.append(sent_data['backward_words'])

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)
        char_id_batch, char_lengths, word_pos_for, word_pos_bak = pad_word_chars_hierarchy(chars_batch)
        char_tags_batch, _, _, _ = pad_word_chars_hierarchy(char_tags_batch, tag=True)

        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']
        feed_dict[model.word_input_ids] = sents_batch
        feed_dict[model.tag_input_ids] = tags_batch
        feed_dict[model.sequence_lengths] = seq_length
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.char_lengths] = char_lengths
        feed_dict[model.word_pos_for] = word_pos_for
        feed_dict[model.word_pos_bak] = word_pos_bak
        feed_dict[model.tag_char_inputs_ids] = char_tags_batch

        # forward_words_batch = pad_tags(for_sents_batch)
        # backward_words_batch = pad_tags(bak_sents_batch)
        # feed_dict[model.forward_words] = forward_words_batch
        # feed_dict[model.backward_words] = backward_words_batch

    elif Model_Parameters['use_hier_char_lm']:
        sents_batch = []
        tags_batch = []
        chars_batch = []
        char_tags_batch = []
        char_lm_forward = []
        char_lm_backward  = []
        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])
            chars_batch.append(sent_data['char_ids'])
            char_lm_forward.append(sent_data['char_lm_forward'])
            char_lm_backward.append(sent_data['char_lm_backward'])

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)
        char_id_batch, char_lengths, word_pos_for, word_pos_bak = pad_word_chars_hierarchy(chars_batch)
        batch_char_lm_for = pad_tags(char_lm_forward)
        batch_char_lm_bak = pad_tags(char_lm_backward)

        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']
        feed_dict[model.word_input_ids] = sents_batch
        feed_dict[model.tag_input_ids] = tags_batch
        feed_dict[model.sequence_lengths] = seq_length
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.char_lengths] = char_lengths
        feed_dict[model.word_pos_for] = word_pos_for
        feed_dict[model.word_pos_bak] = word_pos_bak
        feed_dict[model.char_lm_forward] = batch_char_lm_for
        feed_dict[model.char_lm_backward] = batch_char_lm_bak

    elif Model_Parameters['use_char_alone_model']:
        chars_batch = []
        char_tags_batch = []
        tags_batch = []
        words_batch = []

        for sent_data in sents_idx_data:
            chars_batch.append(sent_data['char_ids'])
            tags_batch.append(sent_data['tag_ids'])
            char_tags_batch.append(sent_data['char_tag_ids'])
            # words_batch.append(sent_data['word_for_char'])

        tags_batch = pad_tags(tags_batch)
        char_id_batch, char_lengths = pad_characters_alone(chars_batch)
        char_tags_batch, _ = pad_characters_alone(char_tags_batch, tag=True)
        # words_batch, _ = pad_characters_alone(words_batch)

        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.sequence_lengths] = char_lengths
        feed_dict[model.tag_input_ids] = char_tags_batch

        # feed_dict[model.word_input_ids] = words_batch
        # feed_dict[model.space_pos] = space_positions
        # feed_dict[model.word_tag_input_ids] = tags_batch
        
    elif Model_Parameters['lm_mode']:
        sents_batch = []
        tags_batch = []
        chars_batch = []
        char_tags_batch = []
        pl_for_sents_batch = []
        pl_bak_sents_batch = []
        nl_for_sents_batch = []
        nl_bak_sents_batch = []
        nl_sequence_pos_batch = []
        nl_sequence_idx_batch = []
        nl_pl_placeholder_idx_batch = []
        pl_sequence_pos_batch = []
        pl_sequence_idx_batch = []
        pl_nl_placeholder_idx_batch = []
        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])
            chars_batch.append(sent_data['char_ids'])
            if Model_Parameters['lm_mode'] == 'pl' or Model_Parameters['lm_mode'] == 'both':
                pl_for_sents_batch.append(sent_data['pl_forward_words'])
                pl_bak_sents_batch.append(sent_data['pl_backward_words'])
                nl_sequence_pos_batch.append(sent_data['nl_sequence_pos'])
                nl_sequence_idx_batch.append(sent_data['nl_sequence_idx'])
                nl_pl_placeholder_idx_batch.append(sent_data['nl_pl_placeholder_idx'])
            if Model_Parameters['lm_mode'] == 'nl' or Model_Parameters['lm_mode'] == 'both':
                nl_for_sents_batch.append(sent_data['nl_forward_words'])
                nl_bak_sents_batch.append(sent_data['nl_backward_words'])
                pl_sequence_pos_batch.append(sent_data['pl_sequence_pos'])
                pl_sequence_idx_batch.append(sent_data['pl_sequence_idx'])
                pl_nl_placeholder_idx_batch.append(sent_data['pl_nl_placeholder_idx'])

        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)
        feed_dict[model.word_input_ids] = sents_batch
        feed_dict[model.tag_input_ids] = tags_batch
        feed_dict[model.sequence_lengths] = seq_length

        char_id_batch, word_lengths = pad_word_chars(chars_batch)
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.word_lengths] = word_lengths

        def pad_positions(sents):
            max_sent_len = max([len(sent) for sent in sents])
            batch_sents = []
            batch_seq_length = []
            for idx, sent in enumerate(sents):
                batch_seq_length.append(len(sent))
                cur_sent = sent + [0] * (max_sent_len - len(sent))
                batch_sents.append([[idx, pos] for pos in cur_sent])
            return batch_sents, batch_seq_length

        def pad_placeholder(placeholder_lengths, type):
            placeholders = []
            max_len = max(placeholder_lengths)
            for length in placeholder_lengths:
                if type == 'pl':
                    placeholders.append([1] * length + [0] * (max_len - length))
                elif type == 'nl':
                    placeholders.append([2] * length + [0] * (max_len - length))
            return placeholders

        if Model_Parameters['lm_mode'] == 'pl' or Model_Parameters['lm_mode'] == 'both':
            pl_for_sents_batch = pad_tags(pl_for_sents_batch)
            pl_bak_sents_batch = pad_tags(pl_bak_sents_batch)
            feed_dict[model.pl_forward_words] = pl_for_sents_batch
            feed_dict[model.pl_backward_words] = pl_bak_sents_batch

            pl_sequence_pos, pl_sequence_lengths = pad_positions(pl_sequence_pos_batch)
            pl_sequence_idx, _ = pad_sentence_words(pl_sequence_idx_batch)
            pl_nl_placeholders_idx, pl_nl_placeholder_lengths = pad_sentence_words(pl_nl_placeholder_idx_batch)
            pl_nl_placeholder_pos = pad_placeholder(pl_nl_placeholder_lengths, 'pl')
            feed_dict[model.pl_sequence_pos] = pl_sequence_pos
            feed_dict[model.pl_sequence_idx] = pl_sequence_idx
            feed_dict[model.pl_sequence_lengths] = pl_sequence_lengths
            feed_dict[model.pl_nl_placeholder_pos] = pl_nl_placeholder_pos
            feed_dict[model.pl_nl_placeholders_idx] = pl_nl_placeholders_idx
            feed_dict[model.pl_nl_placeholder_lengths] = pl_nl_placeholder_lengths

        if Model_Parameters['lm_mode'] == 'nl' or Model_Parameters['lm_mode'] == 'both':
            nl_for_sents_batch = pad_tags(nl_for_sents_batch)
            nl_bak_sents_batch = pad_tags(nl_bak_sents_batch)
            feed_dict[model.nl_forward_words] = nl_for_sents_batch
            feed_dict[model.nl_backward_words] = nl_bak_sents_batch

            nl_sequence_pos, nl_sequence_lengths = pad_positions(nl_sequence_pos_batch)
            nl_sequence_idx, _ = pad_sentence_words(nl_sequence_idx_batch)
            nl_pl_placeholders_idx, nl_pl_placeholder_lengths = pad_sentence_words(nl_pl_placeholder_idx_batch)
            nl_pl_placeholder_pos = pad_placeholder(nl_pl_placeholder_lengths, 'nl')
            feed_dict[model.nl_sequence_pos] = nl_sequence_pos
            feed_dict[model.nl_sequence_idx] = nl_sequence_idx
            feed_dict[model.nl_sequence_lengths] = nl_sequence_lengths
            feed_dict[model.nl_pl_placeholder_pos] = nl_pl_placeholder_pos
            feed_dict[model.nl_pl_placeholders_idx] = nl_pl_placeholders_idx
            feed_dict[model.nl_pl_placeholder_lengths] = nl_pl_placeholder_lengths

    elif Model_Parameters['add_pl_prior']:
        sents_batch = []
        tags_batch = []
        chars_batch = []
        char_tags_batch = []
        prior_batch = []
        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])
            chars_batch.append(sent_data['char_ids'])
            prior_batch.append(sent_data['pl_priors'])

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)

        feed_dict[model.word_input_ids] = sents_batch
        feed_dict[model.tag_input_ids] = tags_batch
        feed_dict[model.sequence_lengths] = seq_length
        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']

        char_id_batch, word_lengths = pad_word_chars(chars_batch)
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.word_lengths] = word_lengths

        prior_batch, _ = pad_sentence_words(prior_batch, type='float')
        feed_dict[model.prob_pl_value] = prior_batch

    elif Model_Parameters['add_keywords']:
        sents_batch = []
        tags_batch = []
        chars_batch = []
        char_tags_batch = []
        keywords_batch = []
        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])
            chars_batch.append(sent_data['char_ids'])
            keywords_batch.append(sent_data['keywords_label'])

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)

        feed_dict[model.word_input_ids] = sents_batch
        feed_dict[model.tag_input_ids] = tags_batch
        feed_dict[model.sequence_lengths] = seq_length
        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']

        char_id_batch, word_lengths = pad_word_chars(chars_batch)
        feed_dict[model.char_input_ids] = char_id_batch
        feed_dict[model.word_lengths] = word_lengths

        keywords_batch, _ = pad_sentence_words(keywords_batch, type='float')
        feed_dict[model.keyword_labels] = keywords_batch

    else:
        sents_batch = []
        tags_batch = []
        chars_batch = []
        char_tags_batch = []
        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])
            if Model_Parameters['use_char_lstm'] or Model_Parameters['use_char_cnn']:
                # cur_char = sent_data['char_ids'] + [[0]] * (max_sent_len - len(sent_data['word_ids']))
                cur_char = sent_data['char_ids']
                chars_batch.append(cur_char)
                if Model_Parameters['add_char_super']:
                    char_tags_batch.append(sent_data['char_tag_ids'])

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)

        feed_dict[model.word_input_ids] = sents_batch
        feed_dict[model.tag_input_ids] = tags_batch
        feed_dict[model.sequence_lengths] = seq_length
        feed_dict[model.dropout_keep_prob] = Model_Parameters['dropout']

        if Model_Parameters['use_char_lstm']:
            char_id_batch, word_lengths = pad_word_chars(chars_batch)
            feed_dict[model.char_input_ids] = char_id_batch
            feed_dict[model.word_lengths] = word_lengths
            if Model_Parameters['add_char_super']:
                char_tags_batch, _ = pad_word_chars(char_tags_batch)
                feed_dict[model.tag_char_inputs_ids] = char_tags_batch

        if Model_Parameters['use_char_cnn']:
            sents_batch, seq_length = pad_sentence_words(sents_batch, Model_Parameters['max_sent_len'])
            tags_batch = pad_tags(tags_batch, Model_Parameters['max_sent_len'])

            feed_dict[model.word_input_ids] = sents_batch
            feed_dict[model.tag_input_ids] = tags_batch
            feed_dict[model.sequence_lengths] = seq_length

            char_id_batch, word_lengths = pad_word_chars(chars_batch, Model_Parameters['max_sent_len'], Model_Parameters['max_char_len'])
            feed_dict[model.char_input_ids] = char_id_batch
            feed_dict[model.word_lengths] = word_lengths

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


def pad_word_chars_hierarchy(chars_batch, tag=False):
    max_sent_len = max([len(sent) for sent in chars_batch])
    max_char_len = max([len(np.concatenate(sent)) for sent in chars_batch])

    char_id_batch = []
    char_lengths = []
    word_pos_for = []
    word_pos_bak = []
    for idx, sent in enumerate(chars_batch):
        chars = np.concatenate(sent)
        char_id_batch.append(np.concatenate((chars, [0]*(max_char_len-len(chars)))))
        if not tag:
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


def TransTags(tag_to_id, two_level=False):
    char_tag = set()
    word2tag_dict = {}
    for word_tag, word_tag_id in tag_to_id.items():
        word2tag_dict.setdefault(word_tag, {})
        if two_level:
            word2tag_dict[word_tag]['B'] = word_tag + '-B'
            word2tag_dict[word_tag]['I'] = word_tag + '-I'
            char_tag.add(word_tag + '-B')
            char_tag.add(word_tag + '-I')
        else:
            word2tag_dict[word_tag]['B'] = word_tag
            word2tag_dict[word_tag]['I'] = word_tag
            char_tag.add(word_tag)
            char_tag.add(word_tag)

    char_tag_to_id = {t: idx for idx, t in enumerate(list(char_tag))}

    return char_tag_to_id, word2tag_dict


def preprocess(line):
    line = re.sub('&lt;', '<', line)
    line = re.sub('&gt;', '>', line)
    line = re.sub('&amp', '&', line)
    line = re.sub('&#xA;', ' ', line)
    line = re.sub('&quot;', '\"', line)
    return line


def tokenization(line, parsing='code'):
    line = preprocess(line)
    line = line.decode('utf8')
    url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    if parsing == 'code':
        # code_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|[^\x00-\x7F]+|[\d.]+|[\w/\-\$%@#\'*.]+|[\"!#$%&(\)*+,-./:;<=>?@[\]^_`{|}~\\]|[^\s\w/\-\$@#'%.]|\.$"
        # return re.findall(code_pattern, line)
        line = re.sub(r'[`]+', ' ', line)
        tknzr = TweetTokenizer()
        return tknzr.tokenize(line)
    elif parsing == 'space':
        code_pattern = r"[^`\s]+|[`]"
        code_tokens = re.findall(code_pattern, line)
        while '`' in code_tokens:
            code_tokens.remove('`')
        return code_tokens


def extract_lm_vocab_pl(train_sents):
    vocab_list = []
    for sent in train_sents:
        vocab_list += [x[0] for x in sent if x[1] == 'B-ENTITY' or x[1] == 'I-ENTITY']
    sorted_vocab = Counter(vocab_list).most_common()
    return sorted_vocab


def extract_lm_vocab_nl(train_sents):
    vocab_list = []
    for sent in train_sents:
        # vocab_list += [x[0] for x in sent if x[1] == 'O']
        vocab_list += [x[0] for x in sent]
    sorted_vocab = Counter(vocab_list).most_common()
    return sorted_vocab


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
