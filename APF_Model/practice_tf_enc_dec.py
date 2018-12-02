import numpy as np
from matplotlib import pyplot as plt
import os
from build_model_multi_variate import *

# create function for generating training samples
input_seq_len = 15
output_seq_len = 20
x = np.linspace(0, 40, 130)
train_data_x = x[:110]


def true_output_signals(x):
    x1 = 2 * np.sin(x)
    x2 = 2 * np.cos(x)
    return x1, x2


def true_input_signals(x):
    x1, x2 = true_output_signals(x)
    y1 = 1.6 * x1 ** 4 - 2 * x2 - 10
    y2 = 1.2 * x2 ** 2 * x1 + 2 * x2 * 3 - x1 * 6
    y3 = 2 * x1 ** 3 + 2 * x2 ** 3 - x1 * x2
    return y1, y2, y3


def noise_func(x, noise_factor=2):
    return np.random.randn(len(x)) * noise_factor


def generate_samples_for_output(x):
    x1, x2 = true_output_signals(x)
    return x1 + noise_func(x1, 0.5), \
           x2 + noise_func(x2, 0.5)


def generate_samples_for_input(x):
    y1, y2, y3 = true_input_signals(x)
    return y1 + noise_func(y1, 2), \
           y2 + noise_func(y2, 2), \
           y3 + noise_func(y3, 2)


def generate_train_samples(x=train_data_x, batch_size=10):
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)

    input_seq_x = [x[i:(i + input_seq_len)] for i in start_x_idx]
    output_seq_x = [x[(i + input_seq_len):(i + input_seq_len + output_seq_len)] for i in start_x_idx]

    input_seq_y = [generate_samples_for_input(x) for x in input_seq_x]
    output_seq_y = [generate_samples_for_output(x) for x in output_seq_x]

    # return shape: (batch_size, time_steps, feature_dims)
    return np.array(input_seq_y).transpose(0, 2, 1), np.array(output_seq_y).transpose(0, 2, 1)


input_seq, output_seq = generate_train_samples(batch_size=100)

# i1, i2, i3= plt.plot(range(input_seq_len), input_seq[0], 'yo', label = 'input_seqs_one_sample')
# o1, o2 = plt.plot(range(input_seq_len,(input_seq_len+output_seq_len)), output_seq[0],
# 'go', label = 'output_seqs_one_sample')
# plt.legend(handles = [i1, o1])
# plt.show()


# Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003
# Network Parameters
# length of input signals
input_seq_len = 15
# length of output signals
output_seq_len = 20
# size of LSTM Cell
hidden_dim = 64
# num of input signals
input_dim = 3
# num of output signals
output_dim = 2
# num of stacked lstm layers
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5


# training process
total_iteractions = 100
batch_size = 16
KEEP_RATE = 0.5
train_losses = []
val_losses = []

x = np.linspace(0, 40, 130)
train_data_x = x[:110]

rnn_model = build_graph(feed_previous=False)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Training losses: ")
    for i in range(total_iteractions):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)

        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:, t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:, t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)

    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'multivariate_ts_model0'))

print("Checkpoint saved at: ", save_path)


# inference process
test_seq_input = np.array(generate_samples_for_input(train_data_x[-15:])).transpose(1, 0)

rnn_model = build_graph(feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    saver = rnn_model['saver']().restore(sess, os.path.join('./', 'multivariate_ts_model0'))

    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1, -1) for t in range(input_seq_len)}
    feed_dict.update(
        {rnn_model['target_seq'][t]: np.zeros([1, output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

    final_preds = np.concatenate(final_preds, axis=0)


# visualize predictions over actual output sequences
test_seq_input = np.array(generate_samples_for_input(train_data_x[-15:])).transpose(1,0)
test_seq_output = np.array(generate_samples_for_output(train_data_x[-20:])).transpose(1,0)
plt.title("Input sequence, predicted and true output sequences")
i1, i2, i3, = plt.plot(range(15), np.array(true_input_signals(x[95:110])).transpose(1, 0), 'c:',
                       label = 'true input sequence')
p1, p2 = plt.plot(range(15, 35), 4 * final_preds, 'ro', label = 'predicted outputs')
t1, t2 = plt.plot(range(15, 35), 4 * np.array(true_output_signals(x[110:])).transpose(1, 0), 'co',
                  alpha = 0.6, label = 'true outputs')
plt.legend(handles = [i1, p1, t1], loc = 'upper left')
plt.show()

plt.title("Predicted and true output sequences")
# i1, i2, i3, = plt.plot(range(15), np.array(true_input_signals(x[95:110])).transpose(1, 0),
# 'c:', label = 'true input sequence')
p1, p2 = plt.plot(range(15, 35), final_preds, 'ro', label = 'predicted outputs')
t1, t2 = plt.plot(range(15, 35), np.array(true_output_signals(x[110:])).transpose(1, 0), 'co', label='true outputs')
plt.legend(handles=[p1, t1], loc='upper left')
plt.show()
