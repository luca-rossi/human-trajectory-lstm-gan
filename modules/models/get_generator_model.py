from keras.models import Model
from keras.layers import Dense, LSTM, Input, Reshape, Concatenate


def get_generator_model(n_features, n_outputs, n_vars):
	start_input = Input(shape=(n_features,))
	latent_input = Input(shape=(n_features,))
	input = Concatenate()([start_input, latent_input])
	reshaped_input = Reshape((int(n_features * 2 / n_vars), n_vars, ))(input)
	out = LSTM(16, stateful=False, return_sequences=False)(reshaped_input)
	out = Dense(32, activation='linear')(out)
	out = Dense(32, activation='linear')(out)
	output = Dense(n_outputs, activation='linear')(out)
	model = Model(inputs=[start_input, latent_input], outputs=output)
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
