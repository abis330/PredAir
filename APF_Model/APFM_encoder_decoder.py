from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import APFM_utils


def define_model(num_encoder_features, num_decoder_features, state_dim):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_features))
    encoder = LSTM(state_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_features))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(state_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_features)
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # plot the model
    plot_model(model, to_file='model.png', show_shapes=True)
    # define encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    # define decoder inference model
    decoder_state_input_h = Input(shape=(state_dim,))
    decoder_state_input_c = Input(shape=(state_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # summarize model
    plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
    plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)
    model.summary()

    return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(inf_enc, inf_dec, input, n_steps, num_output_features):
    # encode
    state = inf_enc.predict(input)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(num_output_features)]).reshape(1, 1, num_output_features)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = inf_dec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)




# configure
num_en_features = 8  # represents number of features of input
num_de_features = 1  # represents number of features of output
state_dimension = 64
tx = 10
ty = 1


train, encoder_inference, decoder_inference = define_model(num_en_features, num_de_features, 128)
train.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01), loss='mean_squared_error')
# generate training dataset
# X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
train_x1, train_x2, train_y, test_x1, test_x2, test_y, minimax_scaler = APFM_utils.get_data_enc_dec('pollution.csv', 8, tx, ty)
# train model
history = train.fit([train_x1, train_x2], train_y, epochs=50, batch_size=72,
                    validation_data=([test_x1, test_x2], test_y), verbose=2, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('error_plot_enc_dec.png')


num_training_points = test_x1.shape[0]
yhat = list()
for i in range(test_x1.shape[0]):
    target = predict_sequence(encoder_inference, decoder_inference,
                              np.expand_dims(test_x1[i], axis=0), 1, num_de_features)
    yhat.append(np.squeeze(target, axis=1))
yhat = np.asarray(yhat)

test_x1 = test_x1.reshape((test_x1.shape[0], test_x1.shape[2] * test_x1.shape[1]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_x1[:, -7:]), axis=1)
inv_yhat = minimax_scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x1[:, -7:]), axis=1)
inv_y = minimax_scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
