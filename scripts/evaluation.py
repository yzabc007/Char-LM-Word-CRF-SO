import tensorflow as tf
import numpy as np
import time

from utils import *
from utils_train import *
from conlleval_py import *


def evaluation(sess, model, Model_Parameters, idx_data):
    tp_chunk_all = 0
    gold_chunks_all = 0
    pred_chunks_all = 0
    data_count = 0
    start = time.time()
    batch_count = 0
    total_loss = 0
    char_loss = 0
    word_loss = 0

    # collect all samples for returning
    pred_all_set = []
    true_all_set = []
    while data_count < len(idx_data):

        batch_data = []
        batch_count += 1
        for i in range(Model_Parameters['batch_size']):
            index = i + data_count
            if index >= len(idx_data):
                continue
                # index %= len(idx_data)
            batch_data.append(idx_data[index])

        data_count += Model_Parameters['batch_size']

        feed_dict_ = get_feed_dict(model, Model_Parameters, batch_data)
        pred_tags, loss_list = predict(model, sess, feed_dict_)

        total_loss += sum(loss_list)
        if len(loss_list) > 1:
            char_loss += loss_list[0]
            word_loss += loss_list[1]

        for idx in xrange(len(pred_tags)):
            seq_length = feed_dict_[model.sequence_lengths][idx]
            y_real = feed_dict_[model.tag_input_ids][idx][:seq_length]
            y_pred = pred_tags[idx]
            true_all_set.append(y_real)
            pred_all_set.append(y_pred)
            assert(len(y_real) == len(y_pred))

            tp_chunk_batch, gold_chunk_batch, pred_chunk_batch = eval_conll(y_real, y_pred, Model_Parameters['id_to_word_tag'])

            tp_chunk_all += tp_chunk_batch
            gold_chunks_all += gold_chunk_batch
            pred_chunks_all += pred_chunk_batch
            # majority_votes(feed_dict_[model.word_input_ids][idx], y_pred, feed_dict_[mode.space_pos][idx])

    prec = 0 if pred_chunks_all == 0 else 1. * tp_chunk_all / pred_chunks_all
    recl = 0 if gold_chunks_all ==0 else 1. * tp_chunk_all / gold_chunks_all
    f1 = 0 if prec + recl == 0 else (2. * prec * recl) / (prec + recl)
    cost_time = time.time() - start
    print 'precision: %6.2f%%' % (100.*prec),
    print 'recall: %6.2f%%' % (100.*recl),
    print 'f1 score: %6.2f%%' % (100.*f1),
    print 'cost time: %i' % cost_time,
    print 'total loss: %6.6f' % (total_loss/batch_count),
    print 'char loss: %6.6f' % (char_loss/batch_count),
    print 'word loss: %6.6f' % (word_loss/batch_count)
    return f1, total_loss/batch_count, true_all_set, pred_all_set


def evaluation_savedmodel(Model_Parameters, test_idx_data):
    print 'Using saved model to predict.'
    data_count = 0
    tp_chunk_all = 0
    gold_chunks_all = 0
    pred_chunks_all = 0

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # begin training
    with tf.Session(config=config) as sess:
        new_saver = tf.train.import_meta_graph(Model_Parameters['restore_mode_path'] + 'model.ckpt-' + Model_Parameters['saved_epoch'] + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(Model_Parameters['restore_mode_path']))

        graph = tf.get_default_graph()

        logits_name = graph.get_tensor_by_name('output_layers/logits:0')
        transition_params_name = graph.get_tensor_by_name('loss/transitions:0')

        word_input_ids = graph.get_tensor_by_name('word_input:0')
        tag_input_ids = graph.get_tensor_by_name('tag_input:0')
        sequence_lengths = graph.get_tensor_by_name('sequence_lengths:0')
        dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        char_input_ids = graph.get_tensor_by_name('char_embedding/char_input:0')
        word_lengths = graph.get_tensor_by_name('char_embedding/word_lengths:0')

        pred_all_set = []
        true_all_set = []

        while data_count <= len(test_idx_data):
            batch_data = []

            print data_count

            for i in range(Model_Parameters['batch_size']):
                index = i + data_count
                if index >= len(test_idx_data):
                    break
                batch_data.append(test_idx_data[index])

            data_count += Model_Parameters['batch_size']

            sents_batch = []
            tags_batch = []
            chars_batch = []
            char_tags_batch = []
            feed_dict_ = {}
            for sent_data in batch_data:
                sents_batch.append(sent_data['word_ids'])
                tags_batch.append(sent_data['tag_ids'])
                chars_batch.append(sent_data['char_ids'])

            sents_batch, seq_length = pad_sentence_words(sents_batch)
            tags_batch = pad_tags(tags_batch)
            feed_dict_[word_input_ids] = sents_batch
            feed_dict_[tag_input_ids] = tags_batch
            feed_dict_[sequence_lengths] = seq_length

            char_id_batch, word_length = pad_word_chars(chars_batch)
            feed_dict_[char_input_ids] = char_id_batch
            feed_dict_[word_lengths] = word_length
            feed_dict_[dropout_keep_prob] = 1.0

            logits, transition_params = sess.run([logits_name, transition_params_name], feed_dict=feed_dict_)

            for idx_batch in xrange(len(seq_length)):
                seq_score = logits[idx_batch,:seq_length[idx_batch],:]
                pred_tags, pred_tags_score = tf.contrib.crf.viterbi_decode(seq_score, transition_params)
                cur_seq_length = feed_dict_[sequence_lengths][idx_batch]
                y_real = feed_dict_[tag_input_ids][idx_batch][:cur_seq_length]
                y_pred = pred_tags
                true_all_set.append(y_real)
                pred_all_set.append(y_pred)
                # print y_real
                # print y_pred
                assert(len(y_real) == len(y_pred))
                tp_chunk_batch, gold_chunk_batch, pred_chunk_batch = eval_conll(y_real, y_pred, Model_Parameters['id_to_word_tag'])
                tp_chunk_all += tp_chunk_batch
                gold_chunks_all += gold_chunk_batch
                pred_chunks_all += pred_chunk_batch

        prec = 0 if pred_chunks_all == 0 else 1. * tp_chunk_all / pred_chunks_all
        recl = 0 if gold_chunks_all ==0  else 1. * tp_chunk_all / gold_chunks_all
        f1 = 0 if prec + recl == 0 else (2. * prec * recl) / (prec + recl)
        print 'Evaluation f1 score: %2.4f' % (f1)

    return pred_all_set
