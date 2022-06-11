from keras.models import Model
from keras.layers import Input, Reshape, Dense, LSTM, Dropout


def get_lstm_model(n_features, n_outputs, n_neurons, dropout, n_vars):
	input = Input(shape=(n_features,))
	reshaped_input = Reshape((int(n_features / n_vars), n_vars, ))(input)
	output = LSTM(n_neurons[0], stateful=False, return_sequences=False)(reshaped_input)
	for i in range(1, len(n_neurons) - 1):
		output = Dense(n_neurons[i], activation='relu')(output)
		output = Dropout(dropout[i - 1])(output)
	if len(n_neurons) > 1:
		output = Dense(n_neurons[-1], activation='relu')(output)
		output = Dropout(dropout[-1])(output)
	output = Dense(n_outputs)(output)
	model = Model(inputs=input, outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
