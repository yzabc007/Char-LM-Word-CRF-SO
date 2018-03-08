import tensorflow as tf
from tensorflow.contrib import rnn
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
        self.tag_char_inputs_ids = tf.placeholder(tf.int32, [None, None], name='chatr_tag_input')

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

            self.embedded_words = tf.nn.embedding_lookup(self.W_word, self.word_input_ids,name='embedded_words')
            self.embedded_words = tf.nn.dropout(self.embedded_words, self.dropout_keep_prob)

            self.batch_size = tf.shape(self.embedded_words)[0]
            self.max_sent_len = tf.shape(self.embedded_words)[1]

            # self.word_embedding_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # char embedding
        with tf.variable_scope('char_embedding') as vs:

            self.W_char = tf.Variable(tf.random_uniform([self.char_vocab_size, self.char_input_dim], -0.25, 0.25),
                trainable=True, name='Char_embedding', dtype=tf.float32)

            # (batch_size, max_char_len, char_input_dim)
            self.embedded_chars = tf.nn.embedding_lookup(self.W_char, self.char_input_ids, name='embedded_chars')
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_keep_prob)

            self.max_char_len = tf.shape(self.embedded_chars)[1]

            # self.char_embedding_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # lower level
        with tf.variable_scope('char_lstm') as vs:
            # (batch_size, max_char_len (whole sentence))
            # s = tf.shape(self.embedded_chars)
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
            # generater inputs to next level
            # seq_fw: (batch_size, max_char_len, char_hidden_dim)
            # need gather vector according to word_positions
            # need padding to max_sent_len
            word_fw_slices = tf.gather_nd(seq_fw, self.word_pos_for, name='word_seg_fw')
            word_bw_slices = tf.gather_nd(seq_bw, self.word_pos_bak, name='word_seg_bw')
            # print 'word_bw_slices', word_bw_slices.get_shape()
            word_slices = tf.concat([word_fw_slices, word_bw_slices], axis=-1, name='word_seg_concat')
            if self.params['connection_method'] == '1':
                self.embedded_words = tf.concat([self.embedded_words, word_slices], axis=-1)
                print 'word_slice: ', word_slices.get_shape()

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            # self.char_level_outputs_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)


        with tf.variable_scope('char_softmax') as vs:
            # add supervision
            if self.params['add_char_super']:
                char_super_output = tf.concat([seq_fw, seq_bw], axis=-1, name='char_super_bilstm')
                char_super_output = tf.reshape(char_super_output, shape=[-1, 2*self.char_hidden_dim])
                # char_super_output: (batch_size*max_char_len, 2*char_hidden_dim)
                W_char_out = tf.get_variable('softmax_char_W',
                    shape=[2*self.char_input_dim, self.char_tag_size], initializer=tf.contrib.layers.xavier_initializer())
                b_char_out = tf.Variable(tf.constant(0.0, shape=[self.char_tag_size]), name='softmax_char_b')
                char_pred = tf.nn.xw_plus_b(char_super_output, W_char_out, b_char_out, name='softmax_char_out')
                self.char_logits = tf.reshape(char_pred, shape=[self.batch_size, self.max_char_len, self.char_tag_size])

                # add logits as inputs
                F_char_logits = tf.gather_nd(self.char_logits, self.word_pos_bak, name='first_char_logit')
                E_char_logits = tf.gather_nd(self.char_logits, self.word_pos_for, name='end_char_logit')
                char_logits_inputs = tf.concat([F_char_logits, E_char_logits], axis=-1, name='combined_logits')
                # print 'char_logits_inputs: ', char_logits_inputs.get_shape()

                if self.params['connection_method'] == '2':
                    # print 'char_logits_inputs', char_logits_inputs.get_shape()
                    self.embedded_words = tf.concat([self.embedded_words, char_logits_inputs], axis=-1)

            # self.char_softmax_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('char_loss') as vs:
            if self.params['add_char_super']:
                char_log_likelihood, self.char_trainsition_params = tf.contrib.crf.crf_log_likelihood(
                    self.char_logits, self.tag_char_inputs_ids, self.char_lengths)
                self.char_loss = tf.reduce_mean(-char_log_likelihood, name='char_crf_loss')
            else:
                self.char_loss = 0

            # self.char_loss_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # with tf.variable_scope('word_softmax') as vs:
        #     self.soft_dim = self.word_input_dim + 2 * self.char_hidden_dim
        #     W_out = tf.get_variable("softmax_W", shape=[self.soft_dim, self.tag_size],initializer=tf.contrib.layers.xavier_initializer())
        #     b_out = tf.Variable(tf.constant(0.0, shape=[self.tag_size]), name='softmax_b')
        #     # (batch_size*max_sent_len, 2*word_hidden_dim)
        #     self.biLSTM_output = tf.reshape(self.embedded_words, [self.batch_size*self.max_sent_len, -1])
        #     # self.predictions = tf.matmul(self.biLSTM_output, W_out) + b_out
        #     self.predictions = tf.nn.xw_plus_b(self.biLSTM_output, W_out, b_out, name='softmax_output')
        #     # (batch_size, max_sent_len, tag_size)
        #     self.word_logits = tf.reshape(self.predictions, [self.batch_size, self.max_sent_len, self.tag_size], name='logits')
        #
        #     self.word_softmax_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        #     print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        #
        # with tf.variable_scope('word_loss') as vs:
        #     log_likelihood, self.word_transition_params = tf.contrib.crf.crf_log_likelihood(
        #         self.word_logits, self.tag_input_ids, self.sequence_lengths)
        #     self.word_loss = tf.reduce_mean(-log_likelihood, name='crf_negloglik_loss')
        #     self.word_loss_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        #     print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

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

            self.word_bilstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
            # self.biLSTM_output = tf.concat([output_seq_fw, output_seq_bw, word_slices, char_logits_inputs], axis=-1, name='BiLSTM')

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('word_softmax') as vs:
            W_out = tf.get_variable("softmax_W", shape=[self.soft_dim, self.tag_size],initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.0, shape=[self.tag_size]), name='softmax_b')
            # (batch_size*max_sent_len, 2*word_hidden_dim)
            self.biLSTM_output = tf.reshape(self.biLSTM_output, [self.batch_size*self.max_sent_len, -1])
            # self.predictions = tf.matmul(self.biLSTM_output, W_out) + b_out
            self.predictions = tf.nn.xw_plus_b(self.biLSTM_output, W_out, b_out, name='softmax_output')
            # (batch_size, max_sent_len, tag_size)
            self.word_logits = tf.reshape(self.predictions, [self.batch_size, self.max_sent_len, self.tag_size], name='logits')

            self.word_softmax_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('word_loss') as vs:
            log_likelihood, self.word_transition_params = tf.contrib.crf.crf_log_likelihood(
                self.word_logits, self.tag_input_ids, self.sequence_lengths)
            self.word_loss = tf.reduce_mean(-log_likelihood, name='crf_negloglik_loss')
            # self.word_loss_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('total_loss') as vs:
            self.total_loss = self.word_loss + self.char_loss

        # for var in tf.trainable_variables():
        #     print var.get_shape()
        #     print var.name

        '''
        --- word_embedding_variables
        word_embedding/Word_embedding:0
        --- char_embedding_variables
        char_embedding/Char_embedding:0
        --- char_lstm_variables
        char_lstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0
        char_lstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0
        char_lstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0
        char_lstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0
        --- char_softmax_variables
        char_softmax/softmax_char_W:0
        char_softmax/softmax_char_b:0
        --- char_loss_variables
        char_loss/transitions:0
        --- word_bilstm_variables
        word_bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0
        word_bilstm/bidirectional_rnn/fw/basic_lstm_cell/bias:0
        word_bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0
        word_bilstm/bidirectional_rnn/bw/basic_lstm_cell/bias:0
        --- word_softmax_variables
        word_softmax/softmax_W:0
        word_softmax/softmax_b:0
        --- word_loss_variables
        word_loss/transitions:0
        '''

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
            grads, gnorm, tf.clip_by_global_norm(grads, self.clip_norm)
            self.char_train_op = optimizer_char.apply_gradients(zip(grads, vs))

            # word loss
            if self.params['frozen']:
                grads, vs = zip(*optimizer_word.compute_gradients(self.word_loss,
                    var_list=self.word_embedding_variables + self.word_bilstm_variables + self.word_softmax_variables + self.word_loss_variables))
            else:
                grads, vs = zip(*optimizer_word.compute_gradients(self.word_loss))
            grads, gnorm, tf.clip_by_global_norm(grads, self.clip_norm)
            self.word_train_op = optimizer_word.apply_gradients(zip(grads, vs))

            # total loss
            grads, vs = zip(*optimizer_total.compute_gradients(self.total_loss))
            grads, gnorm, tf.clip_by_global_norm(grads, self.clip_norm)
            self.total_train_op = optimizer_total.apply_gradients(zip(grads, vs))
        else:
            # self.char_train_op = optimizer_char.minimize(self.char_loss)
            self.word_train_op = optimizer_word.minimize(self.word_loss)
            self.total_train_op = optimizer_total.minimize(self.total_loss)

        print 'Model built!'
        return
