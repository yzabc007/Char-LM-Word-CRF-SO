import tensorflow as tf
from tensorflow.contrib import rnn


class EmbeddingLayer(object):
    def __init__(self, scope_name, vocab_size, embed_dim, trainable):
        self.scope_name = scope_name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.trainable = trainable

    def link(self, inputs, pre_train, embeddings=None):
        with tf.variable_scope(self.scope_name) as vs:
            if pre_train:
                var_embed = tf.Variable(embeddings,
                                        name=self.scope_name + '_var',
                                        dtype=tf.float32,
                                        trainable=self.trainable)
            else:
                assert(embeddings == None)
                var_embed = tf.get_variable(name=self.scope_name + '_var',
                                            dtype=tf.float32,
                                            shape=[self.vocab_size, self.embed_dim],
                                            trainable=self.trainable)

            embedded_inputs = tf.nn.embedding_lookup(var_embed, inputs, name=self.scope_name + '_output')

            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return embedded_inputs


class CharLSTMLayer_Word(object):
    def __init__(self, scope_name, num_layers, input_dim, hidden_dim, dropout):
        self.scope_name = scope_name
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def link(self, embedded_chars, word_lengths):
        with tf.variable_scope(self.scope_name) as vs:
            s = tf.shape(embedded_chars)
            # (batch_size*max_sent_len, max_char_len, char_input_dim)
            new_lstm_embedded_chars = tf.reshape(embedded_chars, shape=[s[0]*s[1], s[2], self.input_dim])
            real_word_lengths = tf.reshape(word_lengths, shape=[s[0]*s[1]])
            if self.num_layers > 1:
                fw_cells = []
                bw_cells = []
                for _ in range(self.num_layers):
                    fw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
                    fw_cells.append(fw_cell)
                    bw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)
                    bw_cells.append(bw_cell)
                char_fw_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
                char_bw_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)
            else:
                char_fw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                char_bw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)

            (seq_fw, seq_bw), (fw_state_tuples, bw_state_tuples) = \
                tf.nn.bidirectional_dynamic_rnn(char_fw_cell,
                                                char_bw_cell,
                                                new_lstm_embedded_chars,
                                                sequence_length=real_word_lengths,
                                                dtype=tf.float32)

            if self.num_layers > 1:
                char_fw_final_out = fw_state_tuples[-1][1]
                char_bw_final_out = bw_state_tuples[-1][1]
            else:
                char_fw_final_out = fw_state_tuples[1]
                char_bw_final_out = bw_state_tuples[1]

            char_output = tf.concat([char_fw_final_out, char_bw_final_out], axis=-1)
            char_output = tf.reshape(char_output, shape=[s[0], s[1], 2*self.hidden_dim])

            char_hiddens = tf.concat([seq_fw, seq_bw], axis=-1)
            char_hiddens = tf.reshape(char_hiddens, shape=[s[0], s[1], s[2], 2*self.hidden_dim])

            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return char_output, char_hiddens


class CharLSTMLayer_Sent(object):
    def __init__(self, scope_name, hidden_dim):
        self.scope_name = scope_name
        self.hidden_dim = hidden_dim

    def link(self, embedded_chars, char_lengths, word_pos_for, word_pos_bak):
        with tf.variable_scope(self.scope_name) as vs:
            # (batch_size, max_char_len (whole sentence))
            char_fw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
            char_bw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
            (seq_fw, seq_bw), ((_, char_fw_final_out), (_, char_bw_final_out)) = \
                tf.nn.bidirectional_dynamic_rnn(char_fw_cell,
                                                char_bw_cell,
                                                embedded_chars,
                                                sequence_length=char_lengths,
                                                dtype=tf.float32)
            # generate inputs to next level
            # seq_fw: (batch_size, max_char_len, char_hidden_dim)
            # need gather vector according to word_positions
            word_fw_slices = tf.gather_nd(seq_fw, word_pos_for)
            word_bw_slices = tf.gather_nd(seq_bw, word_pos_bak)
            # print 'word_bw_slices', word_bw_slices.get_shape()
            word_slices = tf.concat([word_fw_slices, word_bw_slices], axis=-1)
            # print 'word_slice: ', word_slices.get_shape()
            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return word_slices, seq_fw, seq_bw


class CharCNNLayer(object):
    def __init__(self, scope_name, input_dim, filter_size, num_filters, max_char_len):
        self.scope_name = scope_name
        self.input_dim = input_dim
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.max_char_len = max_char_len

    def link(self, embedded_chars):
        with tf.variable_scope(self.scope_name) as vs:
            s = tf.shape(embedded_chars)
            # (batch_size*max_sent_len, max_char_len, char_input_dim)
            new_cnn_embedded_chars = tf.reshape(embedded_chars, shape=[s[0]*s[1], s[2], self.input_dim])

            filter_shape = [self.filter_size, self.input_dim, self.num_filters]
            W_filter = tf.get_variable(self.scope_name + "_W_filter",
                                       shape=filter_shape,
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_filter = tf.Variable(tf.constant(0.0, shape=[self.num_filters]),
                                   name=self.scope_name + '_f_filter')

            # input: [batch, in_width, in_channels]
            # filter: [filter_width, in_channels, out_channels]
            conv = tf.nn.conv1d(new_cnn_embedded_chars,
                                W_filter,
                                stride=1,
                                padding="SAME",
                                name='conv1')
            # (batch_size*max_sent_len, out_width, num_filters)
            # print 'conv: ', conv.get_shape()
            # h_conv1 = tf.nn.relu(tf.nn.bias_add(conv, b_filter, name='add bias'))
            h_conv1 = tf.nn.relu(conv + b_filter)
            h_expand = tf.expand_dims(h_conv1, -1)
            # print 'h_expand: ', h_expand.get_shape()
            # (batch_size*max_sent_len, out_width, num_filters, 1)
            h_pooled = tf.nn.max_pool(h_expand,
                                      ksize=[1, self.max_char_len, 1, 1],
                                      strides=[1, self.max_char_len, 1, 1],
                                      padding="SAME",
                                      name='pooled')
            # print 'pooled: ', h_pooled.get_shape()
            # (batch_size*max_sent_len, num_filters, 1)

            char_pool_flat = tf.reshape(h_pooled, [s[0], s[1], self.num_filters])
            # (batch_size, max_sent, num_filters)

            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return char_pool_flat


class WordLSTMLayer(object):
    def __init__(self, scope_name, num_layers, hidden_dim, dropout):
        self.scope_name = scope_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def link(self, embedded_words, sequence_lengths):
        with tf.variable_scope(self.scope_name) as vs:
            if self.num_layers > 1:
                fw_cells = []
                bw_cells = []
                for _ in range(self.num_layers):
                    fw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
                    fw_cells.append(fw_cell)
                    bw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)
                    bw_cells.append(bw_cell)
                word_fw_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
                word_bw_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)
            else:
                word_fw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)
                word_bw_cell = rnn.LSTMCell(self.hidden_dim, state_is_tuple=True)

            (output_seq_fw, output_seq_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                                word_fw_cell,
                                                word_bw_cell,
                                                embedded_words,
                                                sequence_length=sequence_lengths,
                                                dtype=tf.float32)

            # (batch_size, max_sent_len, 2*word_hidden_dim)
            word_biLSTM_output = tf.concat([output_seq_fw, output_seq_bw], axis=-1)
            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return word_biLSTM_output, output_seq_fw, output_seq_bw


class LabelProjectionLayer(object):
    def __init__(self, scope_name, input_dim, output_dim, batch_size, max_sent_len):
        self.scope_name = scope_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len

    def link(self, inputs):
        with tf.variable_scope(self.scope_name) as vs:
            # (batch_size*max_sent_len, 2*word_hidden_dim)
            reshape_inputs = tf.reshape(inputs, [-1, self.input_dim])

            W_fc1 = tf.get_variable(self.scope_name + '_W_1',
                                    shape=[self.input_dim, self.input_dim // 2],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.Variable(tf.constant(0.0, shape=[self.input_dim // 2]),
                                name=self.scope_name + '_b_1')
            o_fc1 = tf.nn.relu(tf.nn.xw_plus_b(reshape_inputs, W_fc1, b_fc1))

            W_out = tf.get_variable(self.scope_name + '_W_out',
                                    shape=[self.input_dim // 2, self.output_dim],
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.0, shape=[self.output_dim]),
                                name=self.scope_name + '_b_out')
            # self.predictions = tf.matmul(self.biLSTM_output, W_out) + b_out
            predictions = tf.nn.xw_plus_b(o_fc1, W_out, b_out)
            # (batch_size, max_sent_len, tag_size)
            logits = tf.reshape(predictions, [self.batch_size, self.max_sent_len, self.output_dim], name='logits')

            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return logits


class LMLayer(object):
    def __init__(self, scope_name, hidden_dim, output_dim, batch_size, max_seq_len):
        self.scope_name = scope_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def link(self, hidden_seq_fw, hidden_seq_bw, forward_labels, backward_labels, seq_lengths):
        lm_loss = 0
        with tf.variable_scope(self.scope_name + '_forward') as vs:
            fw_seq_reshape = tf.reshape(hidden_seq_fw, shape=[-1, self.hidden_dim])
            # W_for_fc1 = tf.get_variable(self.scope_name + '_for_W_1',
            #                             shape=[self.hidden_dim, self.hidden_dim // 2],
            #                             initializer=tf.contrib.layers.xavier_initializer())
            # b_for_fc1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_dim // 2]),
            #                         name=self.scope_name + '_for_b_1')
            # o_for_fc1 = tf.nn.relu(tf.nn.xw_plus_b(fw_seq_reshape, W_for_fc1, b_for_fc1))
            #
            # W_for_out = tf.get_variable(self.scope_name + '_for_W_out',
            #                              shape=[self.hidden_dim // 2, self.output_dim],
            #                              initializer=tf.contrib.layers.xavier_initializer())
            # b_for_out = tf.Variable(tf.constant(0.0, shape=[self.output_dim]),
            #                          name=self.scope_name + '_for_b_out')

            W_for_out = tf.get_variable(self.scope_name + '_for_W_out',
                                        shape=[self.hidden_dim, self.output_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())
            b_for_out = tf.Variable(tf.constant(0.0, shape=[self.output_dim]),
                                    name=self.scope_name + '_for_b_out')
            forward_preds = tf.nn.xw_plus_b(fw_seq_reshape, W_for_out, b_for_out)
            logits_for = tf.reshape(forward_preds, shape=[self.batch_size, self.max_seq_len, self.output_dim])

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_for,
                                                                    labels=forward_labels)
            for_lm_loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(seq_lengths)))
            lm_loss += for_lm_loss
            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        with tf.variable_scope(self.scope_name + '_backward') as vs:
            bak_seq_reshape = tf.reshape(hidden_seq_bw, shape=[-1, self.hidden_dim])
            # W_bak_fc1 = tf.get_variable(self.scope_name + '_bak_W_1',
            #                             shape=[self.hidden_dim, self.hidden_dim // 2],
            #                             initializer=tf.contrib.layers.xavier_initializer())
            # b_bak_fc1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_dim // 2]),
            #                         name=self.scope_name + '_bak_b_1')
            # o_bak_fc1 = tf.nn.relu(tf.nn.xw_plus_b(char_bak_inputs_reshape, W_bak_fc1, b_bak_fc1))
            #
            # W_char_bak = tf.get_variable(self.scope_name + '_bak_W_out',
            #                              shape=[self.hidden_dim // 2, self.output_dim],
            #                              initializer=tf.contrib.layers.xavier_initializer())
            # b_char_bak = tf.Variable(tf.constant(0.0, shape=[self.output_dim]),
            #                          name=self.scope_name + '_bak_b_out')

            W_bak_out = tf.get_variable(self.scope_name + '_bak_W_out',
                                        shape=[self.hidden_dim, self.output_dim],
                                        initializer=tf.contrib.layers.xavier_initializer())
            b_bak_out = tf.Variable(tf.constant(0.0, shape=[self.output_dim]),
                                    name=self.scope_name + '_bak_b_out')
            backward_preds = tf.nn.xw_plus_b(bak_seq_reshape, W_bak_out, b_bak_out)
            logits_bak = tf.reshape(backward_preds, shape=[self.batch_size, self.max_seq_len, self.output_dim])

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bak,
                                                                    labels=backward_labels)
            bak_lm_loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(seq_lengths)))
            lm_loss += bak_lm_loss

            print(vs.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name))

        return lm_loss

