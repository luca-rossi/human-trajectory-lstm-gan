from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation


def get_dense_model(n_features, n_outputs, n_neurons, dropout):
	input = Input(shape=(n_features,))
#	input = Input(batch_shape=(16, n_features,))
	output = Dense(n_neurons[0])(input)
	output = Activation('relu')(output)
	for i in range(1, len(n_neurons)):
		output = Dropout(dropout[i - 1])(output)
		output = Dense(n_neurons[i])(output)
		output = Activation('relu')(output)
	output = Dropout(dropout[-1])(output)
	output = Dense(n_outputs)(output)
	output = Activation('relu')(output)
	model = Model(inputs=input, outputs=output)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
