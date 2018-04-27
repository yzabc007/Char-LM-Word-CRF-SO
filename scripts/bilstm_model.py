import tensorflow as tf
import numpy as np
import math
import time
from tensorflow.contrib import rnn

from base_model import *
from utils import *
from conlleval_py import eval_conll
from nn import EmbeddingLayer, CharLSTMLayer_Word, CharLSTMLayer_Sent, \
    CharCNNLayer, WordLSTMLayer, LabelProjectionLayer, LMLayer

class bilstm(BaseModel):
    '''
    This class implements the following models:
    1. word-lstm-crf model
    2. char-w-lstm-word-lstm-crf model
    3. char-w-cnn-word-lstm-crf model
    4. char-s-lstm-word-crf model
    5. char-s-lstm-crf model (to-do)
    '''
    def __init__(self, parameters, **kwargs):
        super(bilstm, self).__init__(parameters)
        self.params = parameters
        # global
        # self.batch_size = self.params['batch_size']
        self.tag_size = self.params['tag_size']
        self.char_tag_size = self.params['char_tag_size']
        self.clip_norm = self.params['clip_norm']
        # word
        self.vocab_size = self.params['vocab_size']
        self.word_input_dim = self.params['word_input_dim']
        self.word_hidden_dim = self.params['word_hidden_dim']
        # self.word_bidirect = self.params['word_bidirect']
        # character
        self.char_vocab_size = self.params['char_vocab_size']
        self.char_input_dim = self.params['char_input_dim']
        self.char_hidden_dim = self.params['char_hidden_dim']
        # self.char_bidirect = self.params['char_bidirect']
        self.total_loss = 0

    def _add_placeholder(self):
        # place holders
        self.word_input_ids = tf.placeholder(tf.int32, [None, None], name='word_input')
        self.tag_input_ids = tf.placeholder(tf.int32, [None, None], name='tag_input')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        if self.params['char_encode'] and not self.params['use_hier_char']:
            self.char_input_ids = tf.placeholder(tf.int32, [None, None, None], name='char_input')
            self.word_lengths = tf.placeholder(tf.int32, [None, None], name='word_lengths')
            self.max_char_len = tf.shape(self.char_input_ids)[2]
        elif self.params['use_hier_char']:
            self.char_input_ids = tf.placeholder(tf.int32, [None, None], name='char_input')
            self.char_lengths = tf.placeholder(tf.int32, [None], name='char_lengths')
            self.word_pos_for = tf.placeholder(tf.int32, [None, None, 2], name='word_positions_forward')
            self.word_pos_bak = tf.placeholder(tf.int32, [None, None, 2], name='word_positions_backward')
            self.max_char_len = tf.shape(self.char_input_ids)[1]
            if self.params['char_lm']:
                self.char_lm_forward = tf.placeholder(tf.int32, [None, None], name='char_lm_forward')
                self.char_lm_backward = tf.placeholder(tf.int32, [None, None], name='char_lm_inputs')

        if self.params['word_lm']:
            self.forward_words = tf.placeholder(tf.int32, [None, None], name='forward_words')
            self.backward_words = tf.placeholder(tf.int32, [None, None], name='backward_words')

        # dynamic number
        self.batch_size = tf.shape(self.word_input_ids)[0]
        self.max_sent_len = tf.shape(self.word_input_ids)[1]

    def get_feed_dict(self, sents_idx_data, lr=None, dropout=None):
        feed_dict = {}
        sents_batch = []
        tags_batch = []
        for sent_data in sents_idx_data:
            sents_batch.append(sent_data['word_ids'])
            tags_batch.append(sent_data['tag_ids'])

        sents_batch, seq_length = pad_sentence_words(sents_batch)
        tags_batch = pad_tags(tags_batch)
        feed_dict[self.word_input_ids] = sents_batch
        feed_dict[self.tag_input_ids] = tags_batch
        feed_dict[self.sequence_lengths] = seq_length
        feed_dict[self.dropout_keep_prob] = dropout
        feed_dict[self.lr] = lr

        if self.params['char_encode'] and not self.params['use_hier_char']:
            chars_batch = []
            for sent_data in sents_idx_data:
                chars_batch.append(sent_data['char_ids'])
            if self.params['char_encode'] == 'lstm':
                char_id_batch, word_lengths = pad_word_chars(chars_batch)
                feed_dict[self.char_input_ids] = char_id_batch
                feed_dict[self.word_lengths] = word_lengths
            elif self.params['char_encode'] == 'cnn':
                char_id_batch, word_lengths = pad_word_chars(chars_batch, max_char_len=self.params['max_char_len'])
                feed_dict[self.char_input_ids] = char_id_batch
                feed_dict[self.word_lengths] = word_lengths

        if self.params['use_hier_char']:
            chars_batch = []
            for sent_data in sents_idx_data:
                chars_batch.append(sent_data['char_ids'])
            char_id_batch, char_lengths, word_pos_for, word_pos_bak = pad_word_chars_hierarchy(chars_batch)
            feed_dict[self.char_input_ids] = char_id_batch
            feed_dict[self.char_lengths] = char_lengths
            feed_dict[self.word_pos_for] = word_pos_for
            feed_dict[self.word_pos_bak] = word_pos_bak

            if self.params['char_lm']:
                char_lm_forward = []
                char_lm_backward = []
                for sent_data in sents_idx_data:
                    char_lm_forward.append(sent_data['char_lm_forward'])
                    char_lm_backward.append(sent_data['char_lm_backward'])
                batch_char_lm_for = pad_tags(char_lm_forward)
                batch_char_lm_bak = pad_tags(char_lm_backward)
                feed_dict[self.char_lm_forward] = batch_char_lm_for
                feed_dict[self.char_lm_backward] = batch_char_lm_bak

        if self.params['word_lm']:
            for_sents_batch = []
            bak_sents_batch = []
            for sent_data in sents_idx_data:
                for_sents_batch.append(sent_data['forward_words'])
                bak_sents_batch.append(sent_data['backward_words'])
            feed_dict[self.forward_words] = pad_tags(for_sents_batch)
            feed_dict[self.backward_words] = pad_tags(bak_sents_batch)

        return feed_dict

    def _loss_cal(self, logits):
        with tf.variable_scope('loss') as vs:
            if self.params['use_crf_loss']:
                log_likelihood, self.transition_params = \
                    tf.contrib.crf.crf_log_likelihood(logits,
                                                      self.tag_input_ids,
                                                      self.sequence_lengths)
                word_loss = tf.reduce_mean(-log_likelihood, name='crf_negloglik_loss')
                # print self.transition_params.name
            else:
                # add softmax loss
                self.pred_tags = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=self.tag_input_ids)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                word_loss = tf.reduce_mean(losses)

            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))
        return word_loss

    def add_train_op(self, lr_method, lr, loss, clip=-1, momentum=0):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            elif _lr_m == 'momentum':
                optimizer = tf.train.MomentumOptimizer(lr, momentum)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def build(self):
        self._add_placeholder()
        # word embedding
        word_embed_layer = EmbeddingLayer('word_embeddings',
                                          self.vocab_size,
                                          self.word_input_dim,
                                          self.params['fine_tune_w2v'])

        embedded_words = word_embed_layer.link(self.word_input_ids,
                                               self.params['use_word2vec'],
                                               self.params['embedding_initializer'])
        embedded_words = tf.nn.dropout(embedded_words, self.dropout_keep_prob)
        word_vectors = embedded_words
        word_vector_dim = self.word_input_dim

        if self.params['char_encode'] or self.params['use_hier_char']:
            char_embed_layer = EmbeddingLayer('char_embeddings',
                                              self.char_vocab_size,
                                              self.char_input_dim,
                                              trainable=True)
            embedded_chars = char_embed_layer.link(self.char_input_ids,
                                                   pre_train=False)
            embedded_chars = tf.nn.dropout(embedded_chars, self.dropout_keep_prob)

            if self.params['use_hier_char']:
                char_lstm_sent_layer = CharLSTMLayer_Sent('char_lstm_sent',
                                                          self.char_hidden_dim)
                char_output, char_seq_fw, char_seq_bw = char_lstm_sent_layer.link(embedded_chars,
                                                                             self.char_lengths,
                                                                             self.word_pos_for,
                                                                             self.word_pos_bak)
                word_vectors = tf.concat([word_vectors, char_output], axis=-1)
                word_vector_dim += 2*self.char_hidden_dim

                if self.params['char_lm']:
                    char_lm_layer = LMLayer('char_lm',
                                            self.char_hidden_dim,
                                            self.params['char_vocab_size'],
                                            self.batch_size,
                                            self.max_char_len)
                    self.char_lm_loss = char_lm_layer.link(char_seq_fw,
                                                        char_seq_bw,
                                                        self.char_lm_forward,
                                                        self.char_lm_forward,
                                                        self.char_lengths)
                    self.total_loss += self.char_lm_loss

            elif self.params['char_encode'] == 'lstm':
                char_lstm_word_layer = CharLSTMLayer_Word('char_lstm_word',
                                                          self.params['num_layers'],
                                                          self.char_input_dim,
                                                          self.char_hidden_dim,
                                                          self.dropout_keep_prob)
                char_output, char_hiddens = char_lstm_word_layer.link(embedded_chars, self.word_lengths)
                word_vectors = tf.concat([word_vectors, char_output], axis=-1)
                word_vector_dim += 2*self.char_hidden_dim

            elif self.params['char_encode'] == 'cnn':
                char_cnn_layer = CharCNNLayer('char_cnn',
                                              self.char_input_dim,
                                              self.params['filter_size'],
                                              self.params['num_filters'],
                                              self.params['max_char_len'])
                char_output = char_cnn_layer.link(embedded_chars)
                word_vectors = tf.concat([word_vectors, char_output], axis=-1)
                word_vector_dim += self.params['num_filters']

        word_vectors = tf.nn.dropout(word_vectors, self.dropout_keep_prob)
        # can we omit the following step?
        # word encoding
        if self.params['add_word_lstm']:
            word_lstm_layer = WordLSTMLayer('word_lstm',
                                            self.params['num_layers'],
                                            self.word_hidden_dim,
                                            self.dropout_keep_prob)
            word_bilstm_output, word_seq_fw, word_seq_bw = word_lstm_layer.link(word_vectors,
                                                                                self.sequence_lengths)
            word_vector_dim = 2*self.word_hidden_dim

        label_projection_layer = LabelProjectionLayer('label_projection',
                                                      word_vector_dim,
                                                      self.tag_size,
                                                      self.batch_size,
                                                      self.max_sent_len)
        self.logits = label_projection_layer.link(word_bilstm_output)

        # calculate loss
        self.word_loss = self._loss_cal(self.logits)
        self.total_loss += self.word_loss

        if self.params['word_lm']:
            word_lm_layer = LMLayer('word_lm',
                                    self.word_hidden_dim,
                                    self.params['lm_vocab_size'],
                                    self.batch_size,
                                    self.max_sent_len)
            word_lm_loss = word_lm_layer.link(word_seq_fw,
                                              word_seq_bw,
                                              self.forward_words,
                                              self.backward_words,
                                              self.sequence_lengths)
            # word_lm_loss = self._word_lm(word_seq_fw, word_seq_bw)
            self.total_loss += word_lm_loss

        # for tensorboard
        tf.summary.scalar("loss", self.total_loss)

        # Generic functions that add training op and initialize session
        self.add_train_op(self.params['lr_method'],
                          self.lr,
                          self.total_loss,
                          self.params['clip_norm'],
                          self.params['momentum'])
        self.initialize_session() # now self.sess is defined and vars are init

    def run_epoch(self, train_data, val_data, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.params['batch_size']
        num_train_data = len(train_data)
        nbatches = (num_train_data + batch_size - 1) // batch_size
        # prog = Progbar(target=nbatches)

        train_counter = 0
        batch_counter = 0
        # permutation_train_idx = np.random.permutation(num_train)
        # permutation_train_idx = range(num_train)

        # iterate over dataset
        # for i, (words, labels) in enumerate(minibatches(train, batch_size)):
        while train_counter < num_train_data:
            batch_data = []
            for i in range(batch_size):
                idx = i + train_counter
                if idx >= num_train_data:
                    continue
                batch_data.append(train_data[idx])

            train_counter += batch_size
            batch_counter += 1
            fd = self.get_feed_dict(batch_data, lr=self.params['lr_rate'], dropout=self.params['dropout'])
            # print(fd.keys())
            _, train_loss, summary = self.sess.run([self.train_op, self.total_loss, self.merged], feed_dict=fd)
            # prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if batch_counter % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + batch_counter)

        print('Evaluate on val set: ')
        metrics = self.run_evaluate(val_data)
        # msg = " - ".join(["{} {:04.2f}".format(k, v)
        #         for k, v in metrics.items()])
        # self.logger.info(msg)

        return metrics['f1']

    def run_evaluate(self, test_data):
        batch_size = self.params['batch_size']
        data_count = 0
        batch_count = 0
        total_loss = 0
        start = time.time()

        tp_chunk_all = 0
        gold_chunks_all = 0
        pred_chunks_all = 0
        # collect all samples for returning
        pred_all_set = []
        true_all_set = []
        while data_count < len(test_data):
            batch_data = []
            for i in range(batch_size):
                index = i + data_count
                if index >= len(test_data):
                    continue
                batch_data.append(test_data[index])
            data_count += batch_size
            batch_count += 1

            feed_dict_ = self.get_feed_dict(batch_data, dropout=1.0)
            pred_tags, batch_total_loss = self.predict_batch(feed_dict_)
            total_loss += batch_total_loss
            for idx in xrange(len(pred_tags)):
                seq_length = feed_dict_[self.sequence_lengths][idx]
                y_real = feed_dict_[self.tag_input_ids][idx][:seq_length]
                y_pred = pred_tags[idx]
                true_all_set.append(y_real)
                pred_all_set.append(y_pred)
                assert (len(y_real) == len(y_pred))

                tp_chunk_batch, gold_chunk_batch, pred_chunk_batch \
                    = eval_conll(y_real, y_pred, self.params['id_to_word_tag'])

                tp_chunk_all += tp_chunk_batch
                gold_chunks_all += gold_chunk_batch
                pred_chunks_all += pred_chunk_batch

        prec = 0 if pred_chunks_all == 0 else 1. * tp_chunk_all / pred_chunks_all
        recl = 0 if gold_chunks_all == 0 else 1. * tp_chunk_all / gold_chunks_all
        f1 = 0 if prec + recl == 0 else (2. * prec * recl) / (prec + recl)
        cost_time = time.time() - start

        print 'precision: %6.2f%%' % (100. * prec),
        print 'recall: %6.2f%%' % (100. * recl),
        print 'f1 score: %6.2f%%' % (100. * f1),
        print 'cost time: %i' % cost_time,
        print 'total loss: %6.6f' % (total_loss / batch_count)
        # print 'char loss: %6.6f' % (char_loss / batch_count),
        # print 'word loss: %6.6f' % (word_loss / batch_count)

        return {'f1': f1, 'loss': total_loss / batch_count,
                'true_label': true_all_set, 'pred_label': pred_all_set}
        # return f1, total_loss / batch_count, true_all_set, pred_all_set

    def predict_batch(self, feed_dict_):
        pred_tags_batch = []

        if self.params['use_crf_loss']:
            total_loss, logits, transition_params = \
                self.sess.run([self.total_loss, self.logits, self.transition_params], feed_dict=feed_dict_)
            seq_lengths = feed_dict_[self.sequence_lengths]
            for idx_batch in xrange(len(seq_lengths)):
                seq_score = logits[idx_batch, :seq_lengths[idx_batch], :]
                pred_tags, pred_tags_score = tf.contrib.crf.viterbi_decode(seq_score, transition_params)
                pred_tags_batch.append(pred_tags)
        else:
            total_loss, pred_tags = self.sess.run([self.total_loss, self.pred_tags], feed_dict=feed_dict_)
            seq_lengths = feed_dict_[self.sequence_lengths]
            for idx_batch in xrange(len(seq_lengths)):
                pred_tags_batch.append(pred_tags[idx_batch, :seq_lengths[idx_batch]])

        return pred_tags_batch, total_loss
