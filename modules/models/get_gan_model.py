from keras.models import Model
from keras.layers import Concatenate, Input


def get_gan_model(generator, discriminator, n_features, latent_dim):
	discriminator.trainable = False
	start_input = Input(shape=(n_features,))
	latent_input = Input(shape=(latent_dim,))
	g_output = generator([start_input, latent_input])
	d_input = Concatenate()([start_input, g_output])
	d_output = discriminator(d_input)
	model = Model(inputs=[start_input, latent_input], outputs=d_output)
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model
