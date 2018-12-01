# Air-Pollution-Forecasting-using-Recurrent-Neural-Networks
First step towards solving a real-life problem - air pollution forecasting in Delhi, using deep learning

Will implement the following versions of the APF (Air Pollution Forecasting) Model:
1. vanilla RNN encoder-decoder where both the RNNs will be plain RNNs
2. LSTM-RNN encoder-decoder where both the RNNs will be LSTM
3. vanilla RNN encoder - attention-based decoder where both the RNNs will be plain RNNs
4. vanilla RNN bi-directional encoder - attention-based decoder where both the RNNs will be plain RNNs
5. LSTM-RNN encoder - attention-based decoder where both the RNNs will be LSTM
6. LSTM-RNN bi-directional encoder - attention-based decoder where both the RNNs will be LSTM

Will be trying with Bahdanau and Luong attention mechanisms individually

7. Model based on the temporal-based attention where attention is given to tensors across time steps and also values of features of each tensor at every time step using the reference below: 
https://arxiv.org/abs/1809.04206v2 (Shun-Yao Shih, Fan-Keng Sun, Hung-yi Lee, 2018: "Temporal Pattern Attention for Multivariate Time Series Forecasting")

Will implement the model first in Keras and then in TensorFlow
