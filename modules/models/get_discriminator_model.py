from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Dropout


def get_discriminator_model(n_inputs):
	model = Sequential()

	model.add(Dense(256, kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.2))

	model.add(Dense(256))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.2))

	model.add(Dense(128))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.2))

	model.add(Dense(64))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.2))

	model.add(Dense(32))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(0.2))

	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
