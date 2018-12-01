import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# convert series to supervised learning
def _create_sequences(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_data_enc_dec(filename, num_features, tx=1, ty=1):
    minimax_scaler, n_features, n_input_hours, n_obs, test, train_x1, train_y = get_training_data(filename, num_features,
                                                                                                 tx, ty)
    train_x2 = np.copy(np.expand_dims(train_y, axis=1))
    train_x2 = np.insert(train_x2, 0, 0, axis=1)
    train_x2 = train_x2[:, :-1]
    test_x1, test_y = test[:, :n_obs], test[:, -n_features]
    test_x2 = np.copy(np.expand_dims(test_y, axis=1))
    test_x2 = np.insert(test_x2, 0, 0, axis=1)
    test_x2 = test_x2[:, :-1]
    # reshape input to be 3D [samples, timesteps, features]
    train_x1 = train_x1.reshape((train_x1.shape[0], n_input_hours, n_features))
    train_x2 = train_x2.reshape((train_x2.shape[0], 1, 1))
    train_y = train_y.reshape((train_y.shape[0], 1, 1))
    test_x1 = test_x1.reshape((test_x1.shape[0], n_input_hours, n_features))
    test_x2 = test_x2.reshape((test_x2.shape[0], 1, 1))
    test_y = test_y.reshape((test_y.shape[0], 1, 1))

    return train_x1, train_x2, train_y, test_x1, test_x2, test_y, minimax_scaler


def get_data_attention(filename, num_features, tx=1, ty=1):
    minimax_scaler, n_features, n_input_hours, n_obs, test, train_x, train_y = get_training_data(filename, num_features,
                                                                                                 tx, ty)
    test_x, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_x.shape, len(train_x), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_x = train_x.reshape((train_x.shape[0], n_input_hours, n_features))
    train_y = train_y.reshape((train_y.shape[0], 1, 1))
    test_x = test_x.reshape((test_x.shape[0], n_input_hours, n_features))
    test_y = test_y.reshape((test_y.shape[0], 1, 1))

    return train_x, train_y, test_x, test_y, minimax_scaler


def get_training_data(filename, num_features, tx, ty):
    # load dataset
    dataset = pd.read_csv(filename, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    label_encoder = LabelEncoder()
    values[:, 4] = label_encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    minimax_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = minimax_scaler.fit_transform(values)
    # specify the number of lag hours
    n_input_hours = tx
    n_output_hours = ty
    n_features = num_features
    # frame as supervised learning
    reframed = _create_sequences(scaled, n_input_hours, n_output_hours)
    print(reframed.shape)
    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 48
    n_test_hours_end = 365 * 72
    train = values[:n_train_hours, :]
    test = values[n_train_hours: n_test_hours_end, :]
    # split into input and outputs
    n_obs = n_input_hours * n_features
    train_x, train_y = train[:, :n_obs], train[:, -n_features]
    return minimax_scaler, n_features, n_input_hours, n_obs, test, train_x, train_y


def plot_wrt_feature(filename):
    dataset = pd.read_csv(filename, header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.savefig('plot_wrt_features.png')
