import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import math


class bilstm(object):
    def __init__(self, parameters, **kwargs):
        print 'Building model ...'
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
                                         tf.random_uniform([self.char_vocab_size-1, self.char_input_dim], -drange, drange)],
                                         axis=0)
            W_char = tf.Variable(char_initializer,
                                 trainable=True,
                                 name='Char_embedding',
                                 dtype=tf.float32)

            # (batch_size, max_sent_len, max_char_len, char_input_dim)
            embedded_chars = tf.nn.embedding_lookup(W_char, char_input_ids, name='embedded_chars')
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return embedded_chars

    def _char_lstm(self, embedded_chars, word_lengths):
        with tf.variable_scope('char_lstm') as vs:
            s = tf.shape(embedded_chars)
            new_lstm_embedded_chars = tf.reshape(embedded_chars, shape=[s[0]*s[1], s[2], self.char_input_dim])
            # (batch_size*max_sent_len, max_char_len, char_input_dim)
            real_word_lengths = tf.reshape(word_lengths, shape=[s[0]*s[1]])
            if self.params['num_layers'] > 1:
                fw_cells = []
                bw_cells = []
                for _ in range(self.params['num_layers']):
                    fw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
                    fw_cells.append(fw_cell)
                    bw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
                    bw_cells.append(bw_cell)
                char_fw_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
                char_bw_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)
            else:
                char_fw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)
                char_bw_cell = rnn.LSTMCell(self.char_hidden_dim, state_is_tuple=True)

            (seq_fw, seq_bw), (fw_state_tuples, bw_state_tuples) = \
                tf.nn.bidirectional_dynamic_rnn(char_fw_cell,
                                                char_bw_cell,
                                                new_lstm_embedded_chars,
                                                sequence_length=real_word_lengths,
                                                dtype=tf.float32,
                                                swap_memory=True)

            if self.params['num_layers'] > 1:
                char_fw_final_out = fw_state_tuples[-1][1]
                char_bw_final_out = bw_state_tuples[-1][1]
            else:
                char_fw_final_out = fw_state_tuples[1]
                char_bw_final_out = bw_state_tuples[1]
            # print char_fw_final_out.get_shape()
            # print char_bw_final_out.get_shape()
            char_output = tf.concat([char_fw_final_out, char_bw_final_out], axis=-1, name='Char_BiLSTM')
            char_output = tf.reshape(char_output, shape=[s[0], s[1], 2*self.char_hidden_dim]) # (batch_size, max_sent, 2*char_hidden)
            char_hiddens = tf.concat([seq_fw, seq_bw], axis=-1, name='char_hidden_sequence')
            char_hiddens = tf.reshape(char_hiddens, shape=[s[0], s[1], s[2], 2*self.char_input_dim]) # (batch_size*max_sent, max_char, 2*char_hidden)
            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return char_output, char_hiddens

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

        return word_biLSTM_output

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

    def _char_attention_layer(self, embedded_words, char_hiddens, word_lengths):
        # embedded_words: (batch_size, max_sent, word_dim)
        # char_hiddens: (batch_size, max_sent, max_char, char_dim)
        embedded_words = tf.reshape(embedded_words, shape=[self.batch_size*self.max_sent_len, self.word_hidden_dim])
        char_hiddens = tf.reshape(char_hiddens, shape=[self.batch_size*self.max_sent_len, self.max_char_len, 2*self.char_hidden_dim])
        word_legths = tf.reshape(word_lengths, shape=[self.batch_size*self.max_sent_len])
        with tf.variable_scope('char_attention') as vs:
            W_att = tf.get_variable("W_attention",
                                    shape=[self.word_hidden_dim, 2*self.char_hidden_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_att = tf.Variable(tf.constant(0.0, shape=[2*self.char_hidden_dim]), name='b_attention')
            word_att = tf.nn.relu(tf.nn.xw_plus_b(embedded_words, W_att, b_att)) # (batch_size*max_sent, 2*char_dim)
            # print word_att.get_shape()

            def alpha_context(prev, obs):
                cur_word = obs[0] # (400,)
                cur_chars = obs[1] # (?, 400)
                cur_length = obs[2] # ()
                logits = tf.matmul(cur_chars, tf.expand_dims(cur_word, 1)) # (?, 1)
                probs = tf.nn.softmax(logits) # (?, 1)
                denominator = tf.reduce_sum(tf.boolean_mask(tf.transpose(probs), tf.sequence_mask([cur_length]))) # ()
                real_probs = probs[:cur_length] / denominator
                zero_padding = tf.zeros([self.max_char_len - cur_length, 1])
                real_probs = tf.concat([real_probs, zero_padding], axis=0)
                # real_probs = tf.where(cur_input_ids != 0, probs/denominator, tf.zeros(tf.shape(probs))) # (?, 1)
                # tf.cast(cur_input_ids, tf.float32)
                return real_probs

            batch_alphas = tf.scan(fn=alpha_context,
                                   elems=[word_att, char_hiddens, word_legths],
                                   initializer=tf.zeros([self.max_char_len, 1]),
                                   name='attention_alpha') # (batch_size*max_sent, max_char, 1)
            # print batch_alphas.get_shape()
            # batch_alphas = tf.reshape(batch_alphas, shape=[self.batch_size*self.max_sent_len, self.max_char_len, 1])

            context = tf.reduce_sum(tf.multiply(batch_alphas, char_hiddens), reduction_indices=1) # (batch_size*max_sent, 2*char_hidden)

            context = tf.reshape(context, shape=[self.batch_size, self.max_sent_len, 2*self.char_hidden_dim])
            batch_alphas = tf.squeeze(batch_alphas, axis=-1)
            batch_alphas = tf.reshape(batch_alphas, shape=[self.batch_size, self.max_sent_len, self.max_char_len])

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        return context, batch_alphas

    # TODO
    def _word_attention_layer(self, embedded_words, sentence_lengths):
        # embedded_words: (batch_size, max_sent, word_dim)
        word_dim = tf.shape(embedded_words)[-1]

        word_att_cof = 1  # (batch_size, max_len - 1, 1)
        word_contexts = 1  # (batch_size, max_len, max_len - 1, word_dim)
        word_att_vecs = 1  # (batch_size, max_len, word_dim)
        return word_att_vecs

    def build(self):
        self.word_input_ids = tf.placeholder(tf.int32, [None, None], name='word_input')
        self.tag_input_ids = tf.placeholder(tf.int32, [None, None], name='tag_input')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.sequence_lengths = tf.placeholder(tf.int32, [None], name='sequence_lengths')
        self.char_input_ids = tf.placeholder(tf.int32, [None, None, None], name='char_input')
        self.word_lengths = tf.placeholder(tf.int32, [None, None], name='word_lengths')

        # word embedding
        embedded_words = self._word_embedding(self.word_input_ids)
        # if self.params['dropout']:
        #     embedded_words = tf.nn.dropout(embedded_words, self.dropout_keep_prob)
        self.batch_size = tf.shape(embedded_words)[0]
        self.max_sent_len = tf.shape(embedded_words)[1]

        # char embedding
        embedded_chars = self._char_embedding(self.char_input_ids)
        # if self.params['dropout']:
        #     embedded_chars = tf.nn.dropout(embedded_chars, self.dropout_keep_prob)
        self.max_char_len = tf.shape(embedded_chars)[2]

        char_output, char_hiddens = self._char_lstm(embedded_chars, self.word_lengths)
        word_lstm_input = tf.concat([embedded_words, char_output], axis=-1)

        if self.params['char_attention']:
            context, self.batch_alphas = self._char_attention_layer(embedded_words, char_hiddens, self.word_lengths)
            word_lstm_input = tf.concat([word_lstm_input, context], axis=-1)

        if self.params['dropout']:
            word_lstm_input = tf.nn.dropout(word_lstm_input, self.dropout_keep_prob)

        word_bilstm_output = self._word_lstm(word_lstm_input, self.sequence_lengths)

        self.logits = self._label_prediction(word_bilstm_output)

        with tf.variable_scope('loss') as vs:
            if self.params['use_crf_loss']:
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.tag_input_ids, self.sequence_lengths)
                self.word_loss = tf.reduce_mean(-log_likelihood, name='crf_negloglik_loss')
                # print self.transition_params.name
            else:
                # add softmax loss
                self.pred_tags = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.tag_input_ids)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.word_loss = tf.reduce_mean(losses)

            print vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        self.total_loss = self.word_loss

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
        elif self.params['lr_method'].lower() == 'momentum':
            optimizer_total = tf.train.MomentumOptimizer(self.params['lr_rate'], self.params['momentum'])
            optimizer_word = tf.train.MomentumOptimizer(self.params['lr_rate'], self.params['momentum'])
            optimizer_char = tf.train.MomentumOptimizer(self.params['lr_rate'], self.params['momentum'])

        if self.params['clip_norm'] > 0:
            grads, vs = zip(*optimizer_total.compute_gradients(self.total_loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.params['clip_norm'])
            self.total_train_op = optimizer_total.apply_gradients(zip(grads, vs))

            grads, vs = zip(*optimizer_word.compute_gradients(self.word_loss))
            grads, gnorm = tf.clip_by_global_norm(grads, self.params['clip_norm'])
            self.word_train_op = optimizer_word.apply_gradients(zip(grads, vs))
        else:
            self.total_train_op = optimizer_total.minimize(self.total_loss)
            self.word_train_op = optimizer_word.minimize(self.word_loss)

        return
