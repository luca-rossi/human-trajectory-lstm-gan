from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, Reshape
from keras.backend import int_shape

KERNEL_SIZE = 4


def get_conv_model(n_features, n_outputs, n_neurons, dropout):
	input = Input(shape=(n_features,))

	output = Reshape((int_shape(input)[1], 1))(input)
	output = Dense(n_neurons[0], activation='relu')(output)
	for i in range(1, len(n_neurons)):
		output = Dropout(dropout[i - 1])(output)
		output = Conv1D(filters=n_neurons[i], kernel_size=KERNEL_SIZE, activation='relu')(output)
	output = Dropout(dropout[-1])(output)
	output = Flatten()(output)
	output = Dense(n_outputs)(output)
	model = Model(inputs=input, outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
