from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# hyper-parameters
learning_rate = 0.005
lambda_l2_reg = 0.003

# Network Parameters
# length of input signals
input_seq_len = 10
# length of output signals
output_seq_len = 1
# size of LSTM Cell
hidden_dim = 128
# num of input signals
input_dim = 8
# num of output signals
output_dim = 1
# num of stacked lstm layers
num_stacked_layers = 1
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5

df = pd.read_csv('pollution.csv')
print(df.head())

cols_to_plot = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
i = 1
# plot each column
plt.figure(figsize=(10, 12))
for col in cols_to_plot:
    plt.subplot(len(cols_to_plot), 1, i)
    plt.plot(df[col])
    plt.title(col, y=0.5, loc='left')
    i += 1
plt.savefig('APFM_tf_enc_dec_feature_plots.png')

# Fill NA with 0
# print(df.isnull().sum())
df.fillna(0, inplace=True)

# One-hot encode 'cbwd'
# temp = pd.get_dummies(df['wnd_dir'], prefix='wnd_dir')
# df = pd.concat([df, temp], axis=1)
# del df['wnd_dir'], temp
values = df.iloc[:, 1:].values
label_encoder = LabelEncoder()
values[:, 4] = label_encoder.fit_transform(values[:, 4])

minimax_scaler = MinMaxScaler(feature_range=(0, 1))
scaled = minimax_scaler.fit_transform(values)

y_minimax_scaler = MinMaxScaler(feature_range=(0, 1))
y_values = np.expand_dims(values[:, 0], axis=1)
y_scaled = y_minimax_scaler.fit_transform(y_values)
# Split into train and test - I used the last 1 month data as test, but it's up to you to decide the ratio
# df_train = df.iloc[:(-88), :].copy()
# df_test = df.iloc[-88:, :].copy() # -31 * 24

x_train = scaled[:26280, :].copy()
x_test = scaled[35040:, :].copy()
y_train = np.expand_dims(scaled[:26280, 0], axis=1).copy()
y_test = np.expand_dims(scaled[35040:, 0], axis=1).copy()

# x_train = df_train.loc[:, ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain',
#                            'wnd_dir']].values.copy()  # , 'wnd_dir_NW', 'wnd_dir_SE', 'wnd_dir_cv'
# x_test = df_test.loc[:, ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain',
#                          'wnd_dir']].values.copy()
# y_train = df_train['pollution'].values.copy().reshape(-1, 1)
# y_test = df_test['pollution'].values.copy().reshape(-1, 1)


# z-score transform x - not including those one-how columns!
# for i in range(x_train.shape[1] - 4):
#     temp_mean = x_train[:, i].mean()
#     temp_std = x_train[:, i].std()
#     x_train[:, i] = (x_train[:, i] - temp_mean) / temp_std
#     x_test[:, i] = (x_test[:, i] - temp_mean) / temp_std
#
# # z-score transform y
# train_y_mean = y_train.mean()
# train_y_std = y_train.std()
# y_train = (y_train - train_y_mean) / train_y_std
# y_test = (y_test - train_y_mean) / train_y_std
def generate_train_samples(x=x_train, y=y_train, batch_size=10, inp_seq_len=input_seq_len,
                           out_seq_len=output_seq_len):
    total_start_points = len(x) - inp_seq_len - out_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace=False)

    input_batch_idxs = [list(range(i, i + inp_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis=0)

    output_batch_idxs = [list(range(i + inp_seq_len, i + inp_seq_len + out_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis=0)

    return input_seq, output_seq  # in shape: (batch_size, time_steps, feature_dim)


def generate_test_samples(x=x_test, y=y_test, inp_seq_len=input_seq_len, out_seq_len=output_seq_len):
    total_samples = x.shape[0]

    input_batch_idxs = [list(range(i, i + inp_seq_len)) for i in
                        range((total_samples - inp_seq_len - out_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis=0)

    output_batch_idxs = [list(range(i + inp_seq_len, i + inp_seq_len + out_seq_len)) for i in
                         range((total_samples - inp_seq_len - out_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis=0)

    return input_seq, output_seq


def build_graph(feed_previous=False):
    tf.reset_default_graph()

    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    weights = {
        'out': tf.get_variable('Weights_out',
                               shape=[hidden_dim, output_dim],
                               dtype=tf.float32,
                               initializer=tf.glorot_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out',
                               shape=[output_dim],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.)),
    }

    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
            for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
            for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder.
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO")] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'):
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)

        def _rnn_decoder(decoder_inputs,
                         initial_state,
                         cell,
                         loop_function=None,
                         scope=None):
            """RNN decoder for the sequence-to-sequence model.
            Args:
              decoder_inputs: A list of 2D Tensors [batch_size x input_size].
              initial_state: 2D Tensor with shape [batch_size x cell.state_size].
              cell: rnn_cell.RNNCell defining the cell function and size.
              loop_function: If not None, this function will be applied to the i-th output
                in order to generate the i+1-st input, and decoder_inputs will be ignored,
                except for the first element ("GO" symbol). This can be used for decoding,
                but also for training to emulate http://arxiv.org/abs/1506.03099.
                Signature -- loop_function(prev, i) = next
                  * prev is a 2D Tensor of shape [batch_size x output_size],
                  * i is an integer, the step number (when advanced control is needed),
                  * next is a 2D Tensor of shape [batch_size x input_size].
              scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
            Returns:
              A tuple of the form (outputs, state), where:
                outputs: A list of the same length as decoder_inputs of 2D Tensors with
                  shape [batch_size x output_size] containing generated outputs.
                state: The state of each cell at the final time-step.
                  It is a 2D Tensor of shape [batch_size x cell.state_size].
                  (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                   states can be the same. They are different for LSTM cells though.)
            """
            with variable_scope.variable_scope(scope or "rnn_decoder"):
                state = initial_state
                outputs = []
                prev = None
                for i, inp in enumerate(decoder_inputs):
                    if loop_function is not None and prev is not None:
                        with variable_scope.variable_scope("loop_function", reuse=True):
                            inp = loop_function(prev, i)
                    if i > 0:
                        variable_scope.get_variable_scope().reuse_variables()
                    output, state = cell(inp, state)
                    outputs.append(output)
                    if loop_function is not None:
                        prev = output
            return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                               decoder_inputs,
                               cell,
                               feed_previous,
                               dtype=dtypes.float32,
                               scope=None):
            with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
                enc_cell = copy.deepcopy(cell)
                _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
                if feed_previous:
                    return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
                else:
                    return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
            '''Naive implementation of loop function for _rnn_decoder. Transform prev from
            dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
            used as decoder input of next time step '''
            return tf.matmul(prev, weights['out']) + biases['out']

        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            feed_previous=feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]

    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.abs(_y - _Y))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
            loss=loss,
            learning_rate=learning_rate,
            global_step=global_step,
            optimizer='Adam',
            clip_gradients=GRADIENT_CLIPPING)

    saver = tf.train.Saver

    return dict(
        enc_inp=enc_inp,
        target_seq=target_seq,
        train_op=optimizer,
        loss=loss,
        saver=saver,
        reshaped_outputs=reshaped_outputs,
    )


epochs = 10
batch_size = 72
KEEP_RATE = 0.75
train_losses = []
val_losses = []

rnn_model = build_graph(feed_previous=False)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("Training losses: ")
    for i in range(epochs):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)

        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:, t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        train_losses.append(loss_t)
        print(loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'APFM_tf_enc_dec_model'))
print("Checkpoint saved at: ", save_path)

# predict for test data
test_x, test_y = generate_test_samples()
rnn_model = build_graph(feed_previous=True)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    saver = rnn_model['saver']().restore(sess, os.path.join('./', 'APFM_tf_enc_dec_model'))

    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t] for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]:
                      np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
    final_preds = np.concatenate(final_preds, axis=1)
    temp_final_preds = np.expand_dims(np.squeeze(final_preds), axis=1)
    temp_final_preds = y_minimax_scaler.inverse_transform(temp_final_preds)
    temp_test_y = np.expand_dims(np.squeeze(test_y), axis=1)
    temp_test_y = y_minimax_scaler.inverse_transform(temp_test_y)
    # test_y = test_y * train_y_std + train_y_mean
    # final_preds = final_preds * train_y_std + train_y_mean
    print("Test RMSE is: %.3f" % sqrt(np.mean((temp_final_preds - temp_test_y) ** 2)))

# plotting true output sequence and predicted sequence
temp = [test_y[i].reshape(-1) for i in range(0, test_y.shape[0], test_y.shape[1])]
test_y_expand = np.concatenate(temp, axis=0)
final_preds_expand = np.concatenate([final_preds[i].reshape(-1)
                                     for i in range(0, final_preds.shape[0], test_y.shape[1])], axis=0)
# inverse scale
plt.figure()
plt.plot(final_preds_expand[-88:], color='orange', label='predicted')
plt.plot(test_y_expand[-88:], color='blue', label='actual')
plt.title("Predicted vs Actual (last 88 hours of year 2014)")
plt.legend(loc="upper left")
plt.savefig("apfm_tf_enc_dec_pred_vs_act_epochs_{}.png".format(epochs))
