from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import mean_squared_error
from math import sqrt
from model_utils import *
import utils as utils
import numpy as np
from matplotlib import pyplot
#
# m = 10 # number of training examples
# dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
# print(dataset)
# print('\n')
# print(human_vocab)
# print('\n')
# print(machine_vocab)
#
# dataset[:10]

tx = 3
ty = 1
train_x, train_y, test_x, test_y, minimax_scaler = utils.pre_process_data('pollution.csv', 8, tx, ty)
no_features = 8
# print(X)
# print('\n')
# print(Y)
# print('\n')
# print(Xoh)
# print('\n')
# print(Yoh)
# print('\n')
# print("X.shape:", X.shape)
# print("Y.shape:", Y.shape)
# print("Xoh.shape:", Xoh.shape)
# print("Yoh.shape:", Yoh.shape)


# Defined shared layers as global variables
repeator = RepeatVector(tx)
concatenator = Concatenate(axis=-1)
# densor1 = Dense(10, activation="tanh")
densor2 = Dense(1)
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded
# in this notebook
dot_operator = Dot(axes=1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # START CODE HERE
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states
    # "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the
    # "intermediate energies" variable e. (≈1 lines)
    # e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable
    # energies. (≈1 lines)
    energies = densor2(concat)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next
    # (post-attention) LSTM-cell (≈ 1 line)
    context = dot_operator([alphas, a])
    # END CODE HERE

    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
# output_layer = Dense(len(machine_vocab), activation=softmax)
output_layer = Dense(1)


def model(t_x, t_y, n_a, n_s, num_features):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    num_features -- size of the python dictionary "human_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    x = Input(shape=(t_x, num_features))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(x)

    # Step 2: Iterate for Ty steps
    for t in range(t_y):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[x, s0, c0], outputs=outputs)

    return model


# model = model(tx, ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model = model(tx, ty, n_a, n_s, no_features)

# model.summary()

opt = model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
                    loss='mean_squared_error')

s0 = np.zeros((train_x.shape[0], n_s))
c0 = np.zeros((train_x.shape[0], n_s))
outputs = list(train_y.swapaxes(0, 1))

test_s0 = np.zeros((test_x.shape[0], n_s))
test_c0 = np.zeros((test_x.shape[0], n_s))

history = model.fit([train_x, s0, c0], outputs, epochs=100, batch_size=72,
                    validation_data=([test_x, test_s0, test_c0], list(test_y.swapaxes(0, 1))), verbose=2,
                    shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('error_plot.png')


# make a prediction
yhat = model.predict([test_x, test_s0, test_c0])
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2] * test_x.shape[1]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_x[:, -7:]), axis=1)
inv_yhat = minimax_scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x[:, -7:]), axis=1)
inv_y = minimax_scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
#             'March 3rd 2001', '1 March 2001']
# for example in EXAMPLES:
#     source = string_to_int(example, Tx, human_vocab)
#     source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
#     prediction = model.predict([source, s0, c0])
#     prediction = np.argmax(prediction, axis=-1)
#     output = [inv_machine_vocab[int(i)] for i in prediction]
#
#     print("source:", example)
#     print("output:", ''.join(output))
#
# model.summary()

# attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)
# attention_map = plot_attention_map(model, 8, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)
