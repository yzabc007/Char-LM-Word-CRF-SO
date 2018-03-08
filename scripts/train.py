import time
import cPickle

import tensorflow as tf
import numpy as np
import subprocess as sp

from utils import *
from utils_train import *
from loader import *
from conlleval_py import *
from evaluation import *


def train(model, Model_Parameters, train_idx_data, val_idx_data, test_idx_data):
    np.random.seed(Model_Parameters['random_seed'])
    # build the model and return the optimizer
    model.build()

    num_epochs = Model_Parameters['train_epochs']
    batch_size = Model_Parameters['batch_size']
    num_train = len(train_idx_data)
    best_val_f1 = -np.inf
    best_test_f1 = -np.inf
    loss_train_record = []
    loss_val_record = []
    loss_test_record = []
    f1_train_record = []
    f1_val_record = []
    f1_test_record = []

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # begin training
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if Model_Parameters['restore_mode_path']:
            saver.restore(sess, Model_Parameters['restore_mode_path'])
        start = time.time()
        counter_early_stop = 0

        decay_ratio = 0.05
        for epoch in xrange(num_epochs):
            print 'Iteration: ', epoch
            permutation_train_idx = np.random.permutation(num_train)

            improve_flag = False
            total_loss = 0
            char_loss = 0
            word_loss = 0
            tp_chunk_all = 0
            gold_chunks_all = 0
            pred_chunks_all = 0
            batch_count = 0
            train_data_count = 0
            # counting var for freq calculation
            total_loss_freq = 0
            batch_count_freq = 0
            tp_chunk_freq = 0
            gold_chunks_freq = 0
            pred_chunks_freq = 0
            start_epoch = time.time()
            while train_data_count < num_train:
                # get batch
                batch_data = []
                for i in range(batch_size):
                    index = i + train_data_count
                    if index >= len(permutation_train_idx):
                        continue
                        # index %= len(permutation_train_idx)
                    batch_data.append(train_idx_data[permutation_train_idx[index]])

                # increment counting variables
                train_data_count += batch_size
                batch_count += 1
                batch_count_freq += 1
                # get feed_dict
                feed_dict_ = get_feed_dict(model, Model_Parameters, batch_data)
                # training
                sess.run([model.total_train_op], feed_dict=feed_dict_)

                pred_tags, loss_list = predict(model, sess, feed_dict_)

                total_loss += sum(loss_list)
                total_loss_freq += sum(loss_list)
                if len(loss_list) > 1:
                    char_loss += loss_list[0]
                    word_loss += loss_list[1]
                # count chunks
                for idx in xrange(len(pred_tags)):
                    seq_length = feed_dict_[model.sequence_lengths][idx]
                    y_real = feed_dict_[model.tag_input_ids][idx][:seq_length]
                    y_pred = pred_tags[idx]
                    assert(len(y_real) == len(y_pred))
                    tp_chunk_batch, gold_chunk_batch, pred_chunk_batch = eval_conll(y_real, y_pred, Model_Parameters['id_to_word_tag'])

                    tp_chunk_all += tp_chunk_batch
                    gold_chunks_all += gold_chunk_batch
                    pred_chunks_all += pred_chunk_batch

            # things to do between epoch
            prec = 0 if pred_chunks_all == 0 else 1. * tp_chunk_all / pred_chunks_all
            recl = 0 if gold_chunks_all ==0 else 1. * tp_chunk_all / gold_chunks_all
            f1 = 0 if prec + recl == 0 else (2. * prec * recl) / (prec + recl)

            cost_time_total = time.time() - start
            cost_time_epoch = time.time() - start_epoch
            print '\n*****Epoch: %i, precision: %6.2f%%, recall: %6.2f%%, f1 score: %6.2f%%, total cost time: %i, epoch cost time: %i, total loss: %6.6f, char loss: %6.6f, word loss: %6.6f' % (
                    epoch, 100.*prec, 100.*recl, 100.*f1, cost_time_total, cost_time_epoch, total_loss/batch_count, char_loss/batch_count, word_loss/batch_count)
            print '*****Epoch Evaluating on val set: ',
            val_f1, val_loss, val_trues, val_preds = evaluation(sess, model, Model_Parameters, val_idx_data)
            if val_f1 > best_val_f1:
                improve_flag = True
                counter_early_stop = 0
                best_val_f1 = val_f1
                print '*****New best F1 score on val data!'
                if Model_Parameters['save_model_path']:
                    print ' Saving current model to disk ...'
                    saver.save(sess, Model_Parameters['save_model_path'] + 'model.ckpt', global_step=epoch)
            print '*****Evaluting on test set: ',
            test_f1, test_loss, test_trues, test_preds = evaluation(sess, model, Model_Parameters, test_idx_data)
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                print '*****New best F1 score on test data!'
                if Model_Parameters['save_predict_path']:
                    print '**test prediction results saved!'
                    cPickle.dump(test_trues, open(Model_Parameters['save_predict_path'] + 'test_true_results_' + str(round(test_f1, 4)) + '.pkl', 'wb'))
                    cPickle.dump(test_preds, open(Model_Parameters['save_predict_path'] + 'test_pred_results_' + str(round(test_f1, 4)) + '.pkl', 'wb'))

            if Model_Parameters['fig_name']:
                loss_train_record.append(total_loss/batch_count)
                loss_val_record.append(val_loss)
                loss_test_record.append(test_loss)
                f1_train_record.append(100.*f1)
                f1_val_record.append(100.*val_f1)
                f1_test_record.append(100.*test_f1)
                plotting_storing(Model_Parameters, loss_train_record, loss_val_record, loss_test_record,
                               f1_train_record, f1_val_record, f1_test_record)

            if not improve_flag:
                counter_early_stop += 1
                if counter_early_stop >= Model_Parameters['patiences']:
                    print 'Early stopping at iteration: %d!' % (epoch)
                    break
            print '-------------------------------------------------------------'
            # if Model_Parameters['weight_decay']:
            #     if Model_Parameters['lr_method'] == 'sgd' or Model_Parameters['lr_method'] == 'momentum':
            #         model.params['lr_rate'] /= (1 + Model_Parameters['weight_decay'])

    return
