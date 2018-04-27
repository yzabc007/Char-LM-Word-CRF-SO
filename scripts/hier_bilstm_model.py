import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import math


class hier_bilstm(object):
    def __init__(self, parameters, **kwargs):
        print 'building model ...'
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

    def _word_embedding(self, word_input_ids):
        with tf.variable_scope('word_embedding') as vs:
            if self.params['use_word2vec']:
                W_word = tf.get_variable('Word_embedding',
                                          initializer=self.params['embedding_initializer'],
                                          trainable=self.params['fine_tune_w2v'],
                                          dtype=tf.float32)
            else:
                W_word = tf.Variable(tf.random_uniform([self.vocab_size, self.word_input_dim], -0.25, 0.25),
                                          trainable=True,
                                          name='Word_embedding',
                                          dtype=tf.float32)

            embedded_words = tf.nn.embedding_lookup(W_word, word_input_ids,name='embedded_words')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return embedded_words

    def _char_embedding(self, char_input_ids):
        with tf.variable_scope('char_embedding') as vs:
            drange = np.sqrt(6. / (np.sum([self.char_vocab_size-1, self.char_input_dim])))
            char_initializer = tf.concat([tf.zeros([1, self.char_input_dim]),
                                         tf.random_uniform([self.char_vocab_size-1, self.char_input_dim], -0.25, 0.25)],
                                         axis=0)
            W_char = tf.Variable(char_initializer,
                                 trainable=True,
                                 name='Char_embedding',
                                 dtype=tf.float32)

            # (batch_size, max_sent_len, max_char_len, char_input_dim)
            embedded_chars = tf.nn.embedding_lookup(W_char, char_input_ids, name='embedded_chars')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return embedded_chars

    def _char_lstm_sent(self, embedded_chars):
        with tf.variable_scope('char_lstm') as vs:
            # (batch_size, max_char_len (whole sentence))
            # char lstm
            char_fw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
            char_bw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
            (seq_fw, seq_bw), ((_, char_fw_final_out), (_, char_bw_final_out)) = \
                tf.nn.bidirectional_dynamic_rnn(char_fw_cell,
                                                char_bw_cell,
                                                embedded_chars,
                                                sequence_length=self.char_lengths,
                                                dtype=tf.float32)
            # generate inputs to next level
            # seq_fw: (batch_size, max_char_len, char_hidden_dim)
            # need gather vector according to word_positions
            word_fw_slices = tf.gather_nd(seq_fw, self.word_pos_for, name='word_seg_fw')
            word_bw_slices = tf.gather_nd(seq_bw, self.word_pos_bak, name='word_seg_bw')
            # print 'word_bw_slices', word_bw_slices.get_shape()
            word_slices = tf.concat([word_fw_slices, word_bw_slices], axis=-1, name='word_seg_concat')
            # print 'word_slice: ', word_slices.get_shape()
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return word_slices, seq_fw, seq_bw

    def _word_lstm(self, embedded_words, sequence_lengths):
        with tf.variable_scope('word_bilstm') as vs:
            if self.params['num_layers'] > 1:
                fw_cells = []
                bw_cells = []
                for _ in range(self.params['num_layers']):
                    fw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
                    fw_cells.append(fw_cell)
                    bw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
                    bw_cells.append(bw_cell)
                word_fw_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
                word_bw_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)
            else:
                word_fw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)
                word_bw_cell = rnn.LSTMCell(self.word_hidden_dim, state_is_tuple=True)

            (output_seq_fw, output_seq_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                                word_fw_cell,
                                                word_bw_cell,
                                                embedded_words,
                                                sequence_length=sequence_lengths,
                                                dtype=tf.float32,
                                                swap_memory=True)

            # (batch_size, max_sent_len, 2*word_hidden_dim)
            word_biLSTM_output = tf.concat([output_seq_fw, output_seq_bw], axis=-1, name='BiLSTM')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return word_biLSTM_output, output_seq_fw, output_seq_bw

    def _label_prediction(self, word_bilstm_output):
        with tf.variable_scope('output_layers') as vs:
            # (batch_size*max_sent_len, 2*word_hidden_dim)
            reshape_biLSTM_output = tf.reshape(word_bilstm_output, [-1, 2*self.word_hidden_dim])

            W_fc1 = tf.get_variable("softmax_W_fc1",
                                    shape=[2*self.word_hidden_dim, self.word_hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.Variable(tf.constant(0.0, shape=[self.word_hidden_dim]), name='softmax_b_fc1')
            o_fc1 = tf.nn.relu(tf.nn.xw_plus_b(reshape_biLSTM_output, W_fc1, b_fc1))

            W_out = tf.get_variable("softmax_W_out",
                                    shape=[self.word_hidden_dim, self.tag_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.0, shape=[self.tag_size]), name='softmax_b_out')
            # self.predictions = tf.matmul(self.biLSTM_output, W_out) + b_out
            predictions = tf.nn.xw_plus_b(o_fc1, W_out, b_out, name='softmax_output')
            # (batch_size, max_sent_len, tag_size)
            logits = tf.reshape(predictions, [self.batch_size, self.max_sent_len, self.tag_size], name='logits')

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return logits

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

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        return word_loss

    def _char_lm(self, char_seq_fw, char_seq_bw):
        char_loss = 0
        with tf.variable_scope('char_forward_lm') as vs:
            char_for_inputs = char_seq_fw
            char_for_inputs_reshape = tf.reshape(char_for_inputs, shape=[-1, self.char_hidden_dim])
            W_for_fc1 = tf.get_variable('W_for_fc1',
                                        shape=[self.char_hidden_dim, self.char_hidden_dim // 2],
                                        initializer=tf.contrib.layers.xavier_initializer())
            b_for_fc1 = tf.Variable(tf.constant(0.0, shape=[self.char_hidden_dim // 2]), name='b_for_fc1')
            o_for_fc1 = tf.nn.relu(tf.nn.xw_plus_b(char_for_inputs_reshape, W_for_fc1, b_for_fc1), name='o_for_fc1')

            W_char_for = tf.get_variable('softmax_char_for_W',
                                         shape=[self.char_hidden_dim // 2, self.char_vocab_size],
                                         initializer=tf.contrib.layers.xavier_initializer())
            b_char_for = tf.Variable(tf.constant(0.0, shape=[self.char_vocab_size]), name='softmax_char_for_b')
            char_pred_for = tf.nn.xw_plus_b(o_for_fc1, W_char_for, b_char_for, name='softmax_forward')

            self.char_logits_for = tf.reshape(char_pred_for,
                                              shape=[self.batch_size, self.max_char_len, self.char_vocab_size],
                                              name='char_logits_for')

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.char_logits_for,
                                                                    labels=self.char_lm_forward)
            self.char_for_lm_loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(self.char_lengths)))
            char_loss += self.char_for_lm_loss
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('char_backward_lm') as vs:
            char_bak_inputs = char_seq_bw
            char_bak_inputs_reshape = tf.reshape(char_bak_inputs, shape=[-1, self.char_hidden_dim])
            W_bak_fc1 = tf.get_variable('W_bak_fc1', shape=[self.char_hidden_dim, self.char_hidden_dim // 2],
                                        initializer=tf.contrib.layers.xavier_initializer())
            b_bak_fc1 = tf.Variable(tf.constant(0.0, shape=[self.char_hidden_dim // 2]), name='b_bak_fc1')
            o_bak_fc1 = tf.nn.relu(tf.nn.xw_plus_b(char_bak_inputs_reshape, W_bak_fc1, b_bak_fc1), name='o_bak_fc1')

            W_char_bak = tf.get_variable('softmax_char_bak_W', shape=[self.char_hidden_dim // 2, self.char_vocab_size],
                                         initializer=tf.contrib.layers.xavier_initializer())
            b_char_bak = tf.Variable(tf.constant(0.0, shape=[self.char_vocab_size]), name='softmax_char_bak_b')
            char_pred_bak = tf.nn.xw_plus_b(o_bak_fc1, W_char_bak, b_char_bak, name='softmax_backward')
            self.char_logits_bak = tf.reshape(char_pred_bak,
                                              shape=[self.batch_size, self.max_char_len, self.char_vocab_size],
                                              name='char_logits_bak')

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.char_logits_bak,
                                                                    labels=self.char_lm_backward)
            self.char_bak_lm_loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(self.char_lengths)))
            char_loss += self.char_bak_lm_loss
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return char_loss

    def _word_lm(self, word_seq_fw, word_seq_bw):
        lm_loss = 0
        with tf.variable_scope('word_forward_lm') as vs:
            W_forward = tf.get_variable('softmax_for_W',
                                          shape=[self.word_hidden_dim, self.params['lm_vocab_size']],
                                          initializer=tf.contrib.layers.xavier_initializer())
            b_forward = tf.Variable(tf.constant(0.0, shape=[self.params['lm_vocab_size']]), name='softmax_for_b')
            bilstm_for_reshape = tf.reshape(word_seq_fw, shape=[-1, self.word_hidden_dim])
            pred_forward = tf.nn.xw_plus_b(bilstm_for_reshape, W_forward, b_forward, name='softmax_forward')
            logits_forward = tf.reshape(pred_forward,
                                        shape=[self.batch_size, self.max_sent_len, self.params['lm_vocab_size']],
                                        name='logits_forward')
            for_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_forward,
                                                                        labels=self.forward_words)
            forward_lm_loss = tf.reduce_mean(tf.boolean_mask(for_losses, tf.sequence_mask(self.sequence_lengths)))
            lm_loss += forward_lm_loss

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('word_backward_lm') as vs:
            W_backward = tf.get_variable('softmax_bak_W',
                                          shape=[self.word_hidden_dim, self.params['lm_vocab_size']],
                                          initializer=tf.contrib.layers.xavier_initializer())
            b_backward = tf.Variable(tf.constant(0.0, shape=[self.params['lm_vocab_size']]), name='softmax_bak_b')
            bilstm_bak_reshape = tf.reshape(word_seq_bw, shape=[-1, self.word_hidden_dim])
            pred_backward = tf.nn.xw_plus_b(bilstm_bak_reshape, W_backward, b_backward, name='softmax_backward')
            logits_backward = tf.reshape(pred_backward,
                                         shape=[self.batch_size, self.max_sent_len, self.params['lm_vocab_size']],
                                         name='logits_backward')
            bak_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_backward,
                                                                        labels=self.backward_words)
            backward_lm_loss = tf.reduce_mean(tf.boolean_mask(bak_losses, tf.sequence_mask(self.sequence_lengths)))
            lm_loss += backward_lm_loss

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return lm_loss


    def build(self):
        self.word_input_ids = tf.placeholder(tf.int32, [None, None], name='word_input')
        self.tag_input_ids = tf.placeholder(tf.int32, [None, None], name='tag_input')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.char_input_ids = tf.placeholder(tf.int32, [None, None], name='char_input')
        self.char_lengths = tf.placeholder(tf.int32, [None], name='char_lengths')
        self.word_pos_for = tf.placeholder(tf.int32, [None, None, 2], name='word_positions_forward')
        self.word_pos_bak = tf.placeholder(tf.int32, [None, None, 2], name='word_positions_backward')
        # word segementation point:
        '''
        [[[0,1],[0,5],[0,8]],
         [[1,2],[1,5],[1,13]],
         [[2,3],[2,6],[2,19]]]
        '''
        self.tag_char_inputs_ids = tf.placeholder(tf.int32, [None, None], name='chatr_tag_input')
        # dynamic number
        self.batch_size = tf.shape(self.word_input_ids)[0]
        self.max_sent_len = tf.shape(self.word_input_ids)[1]
        self.max_char_len = tf.shape(self.char_input_ids)[1]

        # word embedding
        embedded_words = self._word_embedding(self.word_input_ids)
        if self.params['dropout']:
            embedded_words = tf.nn.dropout(embedded_words, self.dropout_keep_prob)

        # char embedding
        embedded_chars = self._char_embedding(self.char_input_ids)
        if self.params['dropout']:
            embedded_chars = tf.nn.dropout(embedded_chars, self.dropout_keep_prob)
        # char encoding
        char_output, char_seq_fw, char_seq_bw = self._char_lstm_sent(embedded_chars)
        word_lstm_input = tf.concat([embedded_words, char_output], axis=-1)

        # add character lm
        if self.params['char_lm']:
            self.char_lm_forward = tf.placeholder(tf.int32, [None, None], name='char_lm_forward')
            self.char_lm_backward = tf.placeholder(tf.int32, [None, None], name='char_lm_backward')
            self.char_loss = self._char_lm(char_seq_fw, char_seq_bw)
            self.total_loss += self.char_loss

        if self.params['dropout']:
            word_lstm_input = tf.nn.dropout(word_lstm_input, self.dropout_keep_prob)
        # word encoding
        word_bilstm_output, word_seq_fw, word_seq_bw = self._word_lstm(word_lstm_input, self.sequence_lengths)
        # intermediate fc layers
        self.logits = self._label_prediction(word_bilstm_output)
        # calculate loss
        self.word_loss = self._loss_cal(self.logits)
        self.total_loss += self.word_loss

        if self.params['word_lm']:
            self.forward_words = tf.placeholder(tf.int32, [None, None], name='forward_words')
            self.backward_words = tf.placeholder(tf.int32, [None, None], name='backward_words')
            self.word_lm_loss = self._word_lm(word_seq_fw, word_seq_bw)
            self.total_loss += self.word_lm_loss

        # optimization
        if self.params['lr_method'].lower() == 'adam':
            optimizer_total = tf.train.AdamOptimizer(self.params['lr_rate'])
            # optimizer_word = tf.train.AdamOptimizer(self.params['lr_rate'])
            # optimizer_char = tf.train.AdamOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'adagrad':
            optimizer_total = tf.train.AdagradOptimizer(self.params['lr_rate'])
            # optimizer_word = tf.train.AdagradOptimizer(self.params['lr_rate'])
            # optimizer_char = tf.train.AdagradOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'adadelta':
            optimizer_total = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
            # optimizer_word = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
            # optimizer_char = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'sgd':
            optimizer_total = tf.train.GradientDescentOptimizer(self.params['lr_rate'])
            # optimizer_word = tf.train.GradientDescentOptimizer(self.params['lr_rate'])
            # optimizer_char = tf.train.GradientDescentOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'rmsprop':
            optimizer_total = tf.train.RMSPropOptimizer(self.params['lr_rate'])
            # optimizer_word = tf.train.RMSPropOptimizer(self.params['lr_rate'])
            # optimizer_char = tf.train.RMSPropOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'momentum':
            optimizer_total = tf.train.MomentumOptimizer(self.params['lr_rate'], self.params['momentum'])

        if self.params['clip_norm'] > 0:
            grads, vs = zip(*optimizer_total.compute_gradients(self.total_loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.total_train_op = optimizer_total.apply_gradients(zip(grads, vs))
        else:
            self.total_train_op = optimizer_total.minimize(self.total_loss)

        print 'Model built!'
        return
