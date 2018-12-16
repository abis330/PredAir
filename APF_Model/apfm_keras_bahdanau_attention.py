from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sample_model_utils import *
import apfm_keras_bahdanau_attention_utils as utils
import numpy as np
from matplotlib import pyplot


tx = 10
ty = 1
train_x, train_y, valid_x, valid_y, test_x, test_y, minimax_scaler = utils.get_data_attention('pollution.csv', 8, tx, ty)
no_features = 8


# Defined shared layers as global variables
repeator = RepeatVector(tx)
concatenator = Concatenate(axis=-1)
# densor1 = Dense(10, activation="tanh")
densor2 = Dense(1, kernel_initializer="glorot_normal", bias_initializer='zeros')
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


n_a = 128
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state=True, kernel_initializer="glorot_normal", bias_initializer='zeros')
# output_layer = Dense(len(machine_vocab), activation=softmax)
# dropped = Dropout()
output_layer = Dense(1, kernel_initializer="glorot_normal", bias_initializer='zeros')


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

opt = model.compile(optimizer=Adam(lr=0.005), loss='mean_absolute_error')

s0 = np.zeros((train_x.shape[0], n_s))
c0 = np.zeros((train_x.shape[0], n_s))
outputs = list(train_y.swapaxes(0, 1))

test_s0 = np.zeros((test_x.shape[0], n_s))
test_c0 = np.zeros((test_x.shape[0], n_s))
epochs = 10
history = model.fit([train_x, s0, c0], outputs, epochs=epochs, batch_size=72,
                    validation_data=([test_x, test_s0, test_c0], list(test_y.swapaxes(0, 1))), verbose=2,
                    shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('error_plot_bahdanau_attention.png')


# make a prediction
yhat = model.predict([test_x, test_s0, test_c0])
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2] * test_x.shape[1]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_x[:, -7:]), axis=1)
inv_yhat = minimax_scaler.inverse_transform(inv_yhat)
predicted = inv_yhat[-88:, 0]
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x[:, -7:]), axis=1)
inv_y = minimax_scaler.inverse_transform(inv_y)
actual = inv_y[-88:, 0]
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


plt.figure()
plt.plot(predicted, color='orange', label='predicted')
plt.plot(actual, color='blue', label='actual')
plt.title("Predicted vs Actual (last 88 hours of year 2014)")
plt.legend(loc="upper left")
plt.savefig("apfm_keras_bahdanau_attention_pred_vs_act_epochs_{}.png".format(epochs))
