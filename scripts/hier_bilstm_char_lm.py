import tensorflow as tf
from tensorflow.contrib import rnn
import math


class hier_bilstm_char_lm(object):
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
        self.word_bidirect = self.params['word_bidirect']

        # character
        self.char_vocab_size = self.params['char_vocab_size']
        self.char_input_dim = self.params['char_input_dim']
        self.char_hidden_dim = self.params['char_hidden_dim']
        self.char_bidirect = self.params['char_bidirect']

    def build(self):
        self.word_input_ids = tf.placeholder(tf.int32, [None, None], name='word_input')
        self.tag_input_ids = tf.placeholder(tf.int32, [None, None], name='tag_input')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')

        self.char_input_ids = tf.placeholder(tf.int32, [None, None], name='char_input')
        self.char_lengths = tf.placeholder(tf.int32, [None], name='char_lengths')
        self.word_pos_for = tf.placeholder(tf.int32, [None, None, 2], name='word_positions_forward')
        self.word_pos_bak = tf.placeholder(tf.int32, [None, None, 2], name='word_positions_backward')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # word segementation point:
        '''
        [[[0,1],[0,5],[0,8]],
         [[1,2],[1,5],[1,13]],
         [[2,3],[2,6],[2,19]]]
        '''
        self.char_lm_forward = tf.placeholder(tf.int32, [None, None], name='char_lm_forward')
        self.char_lm_backward = tf.placeholder(tf.int32, [None, None], name='char_lm_inputs')

        # self.tag_char_inputs_ids = tf.placeholder(tf.int32, [None, None], name='chatr_tag_input')

        # word embedding
        with tf.variable_scope('word_embedding') as vs:
            if self.params['use_word2vec']:
                self.W_word = tf.get_variable('Word_embedding',
                                              initializer=self.params['embedding_initializer'],
                                              trainable=self.params['fine_tune_w2v'],
                                              dtype=tf.float32)
            else:
                self.W_word = tf.Variable(tf.random_uniform([self.vocab_size, self.word_input_dim], -0.25, 0.25),
                                          trainable=True,
                                          name='Word_embedding',
                                          dtype=tf.float32)
            # word_input_ids: (256, 100)
            self.embedded_words = tf.nn.embedding_lookup(self.W_word, self.word_input_ids, name='embedded_words')
            # embedded_words: (256, 100, 128)
            self.embedded_words = tf.nn.dropout(self.embedded_words, self.dropout_keep_prob)
            self.batch_size = tf.shape(self.embedded_words)[0]
            self.max_sent_len = tf.shape(self.embedded_words)[1]
            # self.word_embedding_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # char embedding
        with tf.variable_scope('char_embedding') as vs:

            self.W_char = tf.Variable(tf.random_uniform([self.char_vocab_size, self.char_input_dim], -0.25, 0.25),
                                      trainable=True,
                                      name='Char_embedding',
                                      dtype=tf.float32)

            # (batch_size, max_char_len, char_input_dim)
            self.embedded_chars = tf.nn.embedding_lookup(self.W_char, self.char_input_ids, name='embedded_chars')
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)

            self.max_char_len = tf.shape(self.embedded_chars)[1]
            # self.char_embedding_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # character LM
        with tf.variable_scope('char_lstm') as vs:
            # (batch_size, max_char_len (whole sentence))
            # char lstm
            char_fw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
            char_bw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
            (seq_fw, seq_bw), ((_, char_fw_final_out), (_, char_bw_final_out)) = \
                tf.nn.bidirectional_dynamic_rnn(char_fw_cell,
                                                char_bw_cell,
                                                self.embedded_chars,
                                                sequence_length=self.char_lengths,
                                                dtype=tf.float32)

            # print 'seq_fw', seq_fw.get_shape()
            # print 'seq_bw', seq_bw.get_shape()
            # self.char_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

            with tf.variable_scope('char_level_outputs') as vs:
                # generate inputs to next level
                # seq_fw: (batch_size, max_char_len, char_hidden_dim)
                # need gather vector according to word_positions
                # need padding to max_sent_len
                word_fw_slices = tf.gather_nd(seq_fw, self.word_pos_for, name='word_seg_fw')
                word_bw_slices = tf.gather_nd(seq_bw, self.word_pos_bak, name='word_seg_bw')
                # print 'word_bw_slices', word_bw_slices.get_shape()
                word_slices = tf.concat([word_fw_slices, word_bw_slices], axis=-1, name='word_seg_concat')
                if self.params['connection_method'] == '1':
                    # input character hidden states to next level
                    self.embedded_words = tf.concat([self.embedded_words, word_slices], axis=-1)
                    print 'word_slice: ', word_slices.get_shape()

                # self.char_level_outputs_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        if self.params['add_char_super']:
            with tf.variable_scope('char_forward_lm') as vs:
                char_for_inputs = seq_fw
                char_for_inputs_reshape = tf.reshape(char_for_inputs, shape=[-1, self.char_hidden_dim])
                W_for_fc1 = tf.get_variable('W_for_fc1', shape=[self.char_hidden_dim, 512], initializer=tf.contrib.layers.xavier_initializer())
                b_for_fc1 = tf.Variable(tf.constant(0.0, shape=[512]), name='b_for_fc1')
                o_for_fc1 = tf.nn.relu(tf.nn.xw_plus_b(char_for_inputs_reshape, W_for_fc1, b_for_fc1), name='o_for_fc1')

                W_char_for = tf.get_variable('softmax_char_for_W', shape=[512, self.char_vocab_size], initializer=tf.contrib.layers.xavier_initializer())
                b_char_for = tf.Variable(tf.constant(0.0, shape=[self.char_vocab_size]), name='softmax_char_for_b')
                char_pred_for = tf.nn.xw_plus_b(o_for_fc1, W_char_for, b_char_for, name='softmax_forward')

                self.char_logits_for = tf.reshape(char_pred_for,
                                                  shape=[self.batch_size, self.max_char_len, self.char_vocab_size],
                                                  name='char_logits_for')

                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.char_logits_for, labels=self.char_lm_forward)
                self.char_for_lm_loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(self.char_lengths)))
                self.char_loss = self.char_for_lm_loss
                print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

            with tf.variable_scope('char_backward_lm') as vs:
                char_bak_inputs = seq_bw
                char_bak_inputs_reshape = tf.reshape(char_bak_inputs, shape=[-1, self.char_hidden_dim])
                W_bak_fc1 = tf.get_variable('W_bak_fc1', shape=[self.char_hidden_dim, 512], initializer=tf.contrib.layers.xavier_initializer())
                b_bak_fc1 = tf.Variable(tf.constant(0.0, shape=[512]), name='b_bak_fc1')
                o_bak_fc1 = tf.nn.relu(tf.nn.xw_plus_b(char_bak_inputs_reshape, W_bak_fc1, b_bak_fc1), name='o_bak_fc1')

                W_char_bak = tf.get_variable('softmax_char_bak_W', shape=[512, self.char_vocab_size], initializer=tf.contrib.layers.xavier_initializer())
                b_char_bak = tf.Variable(tf.constant(0.0, shape=[self.char_vocab_size]), name='softmax_char_bak_b')
                char_pred_bak = tf.nn.xw_plus_b(o_bak_fc1, W_char_bak, b_char_bak, name='softmax_backward')
                self.char_logits_bak = tf.reshape(char_pred_bak,
                                                  shape=[self.batch_size, self.max_char_len, self.char_vocab_size],
                                                  name='char_logits_bak')

                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.char_logits_bak, labels=self.char_lm_backward)
                self.char_bak_lm_loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(self.char_lengths)))
                self.char_loss += self.char_bak_lm_loss
                print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        else:
            self.char_loss = 0

        with tf.variable_scope('word_bilstm') as vs:
            # print 'embedded_words', self.embedded_words.get_shape()
            lstm_fw_cell = rnn.BasicLSTMCell(self.word_hidden_dim, state_is_tuple=True)
            lstm_bw_cell = rnn.BasicLSTMCell(self.word_hidden_dim, state_is_tuple=True)
            (output_seq_fw, output_seq_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                                lstm_bw_cell,
                                                                                self.embedded_words,
                                                                                sequence_length=self.sequence_lengths,
                                                                                dtype=tf.float32)
            # (batch_size, max_sent_len, 2*word_hidden_dim)
            self.biLSTM_output = tf.concat([output_seq_fw, output_seq_bw], axis=-1)
            self.soft_dim = 2 * self.word_hidden_dim

            if self.params['connection_method'] == '3':
                print 'Connect char hidden to word softmax'
                self.biLSTM_output = tf.concat([output_seq_fw, output_seq_bw, word_slices], axis=-1)
                self.soft_dim += 2 * self.char_hidden_dim
            if self.params['connection_method'] == '4':
                print 'Connect char labels to word softmax'
                self.biLSTM_output = tf.concat([output_seq_fw, output_seq_bw, char_logits_inputs], axis=-1)
                self.soft_dim += 2 * self.char_tag_size

            # self.word_bilstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            # self.biLSTM_output = tf.concat([output_seq_fw, output_seq_bw, word_slices, char_logits_inputs], axis=-1, name='BiLSTM')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('word_softmax') as vs:
            W_out = tf.get_variable("softmax_W", shape=[self.soft_dim, self.tag_size], initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.0, shape=[self.tag_size]), name='softmax_b')
            # (batch_size*max_sent_len, 2*word_hidden_dim)
            self.biLSTM_output_reshape = tf.reshape(self.biLSTM_output, [self.batch_size*self.max_sent_len, -1])
            # self.predictions = tf.matmul(self.biLSTM_output, W_out) + b_out
            self.predictions = tf.nn.xw_plus_b(self.biLSTM_output_reshape, W_out, b_out, name='softmax_output')
            # (batch_size, max_sent_len, tag_size)
            self.logits = tf.reshape(self.predictions, [self.batch_size, self.max_sent_len, self.tag_size], name='logits')

            # self.word_softmax_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('word_loss') as vs:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                            self.tag_input_ids,
                                                                                            self.sequence_lengths)
            self.word_loss = tf.reduce_mean(-log_likelihood, name='crf_negloglik_loss')
            # self.word_loss_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('total_loss') as vs:
            if self.params['connection_method'] == '0':
                self.total_loss = self.word_loss
            else:
                self.total_loss = self.word_loss + self.params['lm_lambda'] * self.char_loss

        # optimization
        if self.params['lr_method'].lower() == 'adam':
            optimizer_total = tf.train.AdamOptimizer(self.params['lr_rate'])
            optimizer_word = tf.train.AdamOptimizer(self.params['lr_rate'])
            optimizer_char = tf.train.AdamOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'adagrad':
            optimizer_total = tf.train.AdagradOptimizer(self.params['lr_rate'])
            optimizer_word = tf.train.AdagradOptimizer(self.params['lr_rate'])
            optimizer_char = tf.train.AdagradOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'adadelta':
            optimizer_total = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
            optimizer_word = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
            optimizer_char = tf.train.AdadeltaOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'sgd':
            optimizer_total = tf.train.GradientDescentOptimizer(self.params['lr_rate'])
            optimizer_word = tf.train.GradientDescentOptimizer(self.params['lr_rate'])
            optimizer_char = tf.train.GradientDescentOptimizer(self.params['lr_rate'])
        elif self.params['lr_method'].lower() == 'rmsprop':
            optimizer_total = tf.train.RMSPropOptimizer(self.params['lr_rate'])
            optimizer_word = tf.train.RMSPropOptimizer(self.params['lr_rate'])
            optimizer_char = tf.train.RMSPropOptimizer(self.params['lr_rate'])

        if self.params['clip_norm'] > 0:

            # char loss
            if self.params['frozen']:
                grads, vs = zip(*optimizer_char.compute_gradients(self.char_loss,
                    var_list=self.char_embedding_variables + self.char_lstm_variables + self.char_softmax_variables + self.char_loss_variables))
            else:
                grads, vs = zip(*optimizer_char.compute_gradients(self.char_loss))

            grads, gnorm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.char_train_op = optimizer_char.apply_gradients(zip(grads, vs))

            # word loss
            if self.params['frozen']:
                grads, vs = zip(*optimizer_word.compute_gradients(self.word_loss,
                    var_list=self.word_embedding_variables + self.word_bilstm_variables + self.word_softmax_variables + self.word_loss_variables))
            else:
                grads, vs = zip(*optimizer_word.compute_gradients(self.word_loss))

            grads, gnorm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.word_train_op = optimizer_word.apply_gradients(zip(grads, vs))

            # total loss
            grads, vs = zip(*optimizer_total.compute_gradients(self.total_loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.clip_norm)
            self.total_train_op = optimizer_total.apply_gradients(zip(grads, vs))

        else:

            if self.params['add_char_super']:
                self.char_train_op = optimizer_char.minimize(self.char_loss)

            self.word_train_op = optimizer_word.minimize(self.word_loss)

            self.total_train_op = optimizer_total.minimize(self.total_loss)

        print 'Model built!'

        return
