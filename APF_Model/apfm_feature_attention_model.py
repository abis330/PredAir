import logging
import tensorflow as tf
from tensorflow.contrib import rnn
from apfm_feature_attention_wrapper import TemporalPatternAttentionCellWrapper


class Model:
    def __init__(self, para, data_generator):
        self.data_generator = data_generator
        self.para = para
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.summary_ops = None  # variable storing summary of statistics like loss and accuracy
        self._build_graph()
        if self.para.mode == "train":
            self._build_optimizer()

        self.saver = tf.train.Saver(max_to_keep=self.para.num_epochs)

    @staticmethod
    def _compute_loss(outputs, labels):
        """
        outputs: [batch_size, output_size]
        labels: [batch_size, output_size]
        """
        loss = tf.reduce_mean(
            tf.losses.absolute_difference(
                labels=labels, predictions=outputs))
        return loss

    def _build_graph(self):
        logging.debug("Building graph")

        # rnn_inputs: [batch_size, max_len, input_size]
        # rnn_inputs_len: [batch_size]
        # target_outputs: [batch_size, max_len, output_size]
        self.rnn_inputs, self.rnn_inputs_len, self.target_outputs = \
            self.data_generator.inputs(self.para.mode, self.para.batch_size)

        input_shape = self.rnn_inputs[0].shape
        # rnn_inputs_embed: [batch_size, max_len, num_units]
        with tf.name_scope('embedding'):
            # # This Variable will hold the state of the weights for the layer
            # w1_initializer = tf.initializers.he_normal()
            # w_1 = tf.Variable(w1_initializer([self.para.output_size, self.para.num_units]), name='w1')
            # b_1 = tf.Variable(tf.zeros([self.para.num_units]), name='b1')
            # self.rnn_inputs_embed = tf.nn.relu(tf.matmul(self.rnn_inputs, w_1) + b_1)
            self.rnn_inputs_embed = tf.nn.relu(
                tf.layers.dense(self.rnn_inputs, self.para.num_units, kernel_initializer=tf.initializers.he_normal()))

        # all_rnn_states: [batch_size, max_len, num_units]
        # final_rnn_states: [LSTMStateTuple], len = num_layers
        # LSTMStateTuple: (c: [batch_size, num_units],
        #                  h: [batch_size, num_units])
        self.rnn_inputs_embed = tf.unstack(self.rnn_inputs_embed, axis=1)
        self.all_rnn_states, self.final_rnn_states = rnn.static_rnn(
            cell=self._build_rnn_cell(),
            inputs=self.rnn_inputs_embed,
            sequence_length=self.rnn_inputs_len,
            dtype=self.dtype
        )

        # final_rnn_states: [batch_size, num_units]
        self.final_rnn_states = tf.concat([self.final_rnn_states[i][1] for i in range(self.para.num_layers)], axis=1)

        with tf.name_scope('output'):
            # # This Variable will hold the state of the weights for the layer
            # w2_initializer = tf.initializers.glorot_normal()
            # w_2 = tf.Variable(w2_initializer([self.para.num_units, self.para.output_size]), name='w2')
            # b_2 = tf.Variable(tf.zeros([self.para.output_size]), name='b2')
            # # all_rnn_outputs: [batch_size, output_size]
            # self.all_rnn_outputs = tf.matmul(self.final_rnn_states, w_2) + b_2
            self.all_rnn_outputs = tf.layers.dense(self.final_rnn_states, self.para.output_size)

        if self.para.mode == "train" or self.para.mode == "validation":
            self.labels = self.target_outputs[:, self.para.max_len - 1, :]
            self.loss = Model._compute_loss(outputs=self.all_rnn_outputs, labels=self.labels)
            tf.summary.scalar('{}_loss'.format(self.para.mode), self.loss)
        elif self.para.mode == "test":
            self.labels = self.target_outputs[:, self.para.max_len - 1, :]

        self.summary_ops = tf.summary.merge_all()

    def _build_optimizer(self):
        logging.debug("Building optimizer")

        trainable_variables = tf.trainable_variables()
        # if self.para.decay > 0:
        #     lr = tf.train.exponential_decay(
        #         self.para.learning_rate,
        #         self.global_step,
        #         self.para.decay,
        #         0.995,
        #         staircase=True,
        #     )
        # else:
        #     lr = self.para.learning_rate
        self.opt = tf.train.AdamOptimizer(self.para.learning_rate)
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                                   self.para.max_gradient_norm)
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step,
        )

    def _build_single_cell(self):
        cell = rnn.LSTMBlockCell(self.para.num_units)
        if self.para.mode == "train":
            cell = rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=self.para.keep_prob,
                output_keep_prob=self.para.keep_prob,
                state_keep_prob=self.para.keep_prob,
            )
        cell = TemporalPatternAttentionCellWrapper(cell, self.para.attention_len)
        return cell

    def _build_rnn_cell(self):
        return tf.contrib.rnn.MultiRNNCell(
            [self._build_single_cell() for _ in range(self.para.num_layers)])
