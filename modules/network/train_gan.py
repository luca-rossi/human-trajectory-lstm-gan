import numpy as np
import pandas as pd
from numpy import zeros
from numpy import ones
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils import generate_latent_points

# generate n real samples with class labels
def generate_real_samples(n, dataset):
	idx = np.random.randint(len(dataset), size=n)
	X = dataset[idx, :]
	# generate class labels
	y = ones((n, 1))
	return X, y


def generate_gan_batch(dataset, latent_dim, n_batch):
	# prepare input for gan
	idx = np.random.randint(len(dataset), size=n_batch)
	x_gan_start = dataset[idx, :16]
	# prepare points in latent space as input for the generator
	x_gan_latent = generate_latent_points(latent_dim, n_batch)
	# create inverted labels for the fake samples
	y_gan = ones((n_batch, 1))
	return x_gan_start, x_gan_latent, y_gan


def generate_discriminator_batch(dataset, fake_dataset, n_batch, x_gan_start):
	# prepare real samples
	x_real, y_real = generate_real_samples(n_batch, dataset)
	# prepare fake samples
	idx = np.random.randint(len(fake_dataset), size=n_batch)
	x_fake = fake_dataset[idx, :]
	# create class labels
	y_fake = zeros((n_batch, 1))
	# concatenate and shuffle real and fake data
	x_discriminator = np.concatenate((x_real, np.concatenate((x_gan_start, x_fake), axis=1)))
	y_discriminator = np.concatenate((y_real, y_fake))
	x_discriminator, y_discriminator = shuffle(x_discriminator, y_discriminator)
	return x_discriminator, y_discriminator


def init_fake_dataset(dataset, generator, latent_dim, n_batch):
	idx = np.random.randint(len(dataset), size=n_batch)
	x_gan_start = dataset[idx, :16]
	latent_input = generate_latent_points(latent_dim, n_batch)
	return generator.predict([x_gan_start, latent_input])


def update_fake_dataset(fake_dataset, x_fake, max_fake_size):
	fake_dataset = np.concatenate((fake_dataset, x_fake))
	if len(fake_dataset) > max_fake_size:
		idx = np.random.randint(len(fake_dataset), size=int(max_fake_size / 2))
		fake_dataset = fake_dataset[idx, :]
	return fake_dataset


# train the discriminator model
def train_gan(dataset, g_model, d_model, gan_model, latent_dim, n_epochs=5000, n_batch=512, n_eval=2000):
	MAX_FAKE_SIZE = 100000
	VAL_TRAIN_RATIO = 0.2

	fake_dataset = init_fake_dataset(dataset, g_model, latent_dim, n_batch)

	# run epochs manually
	for i in range(n_epochs):
		# train gan: update the generator via the discriminator's error
		x_gan_start, x_gan_latent, y_gan = generate_gan_batch(dataset, latent_dim, n_batch)
		g_loss = gan_model.train_on_batch([x_gan_start, x_gan_latent], y_gan)# TODO, validation_split=VAL_TRAIN_RATIO)

		# update fake dataset to train the discriminator
		x_fake = g_model.predict([x_gan_start, x_gan_latent])
		fake_dataset = update_fake_dataset(fake_dataset, x_fake, MAX_FAKE_SIZE)

		# train discriminator
		x_discriminator, y_discriminator = generate_discriminator_batch(dataset, fake_dataset, n_batch, x_gan_start)
		d_loss = d_model.train_on_batch(x_discriminator, y_discriminator)# todo, validation_split=VAL_TRAIN_RATIO)


		if i % int(n_epochs / 20) == 0:
			from config import MODEL_PATH
			g_model.save_weights(MODEL_PATH)

			# prepare real samples
			X_real, Y_real = generate_real_samples(n_batch, dataset)
			# evaluate discriminator on real examples
			_, acc_real = d_model.evaluate(X_real, Y_real, verbose=0)

			# prepare fake examples
			x_gan_start = X_real[:, :16]
			# generate points in latent space
			latent_input = generate_latent_points(latent_dim, n_batch)
			# predict outputs
			X_fake = g_model.predict([x_gan_start, latent_input])
			# create class labels
			Y_fake = zeros((n_batch, 1))
			# evaluate discriminator on fake examples
			_, acc_fake = d_model.evaluate(np.concatenate((x_gan_start, X_fake), axis=1), Y_fake, verbose=0)

			'''
			# summarize discriminator performance
			print(i, acc_real, acc_fake, g_loss, d_loss)
			plt.axis([-1, 1, -1, 1])
			for j in range(1):
				X_real_el = X_real[j]
				X_real_el = X_real_el.reshape((int(len(X_real_el) / 2), 2))
				X_real_el = pd.DataFrame(X_real_el, columns=['x', 'y'])
				plt.plot('x', 'y', c='blue', data=X_real_el)
			plt.savefig('plots/gan_e%03d_real.png' % (i + 1))
			plt.show()

			plt.axis([-1, 1, -1, 1])
			for j in range(1):
				X_input_el = x_gan_start[j]
				X_input_el = X_input_el.reshape((int(len(X_input_el) / 2), 2))
				X_input_el = pd.DataFrame(X_input_el, columns=['x', 'y'])

				X_fake_el = X_fake[j]
				X_fake_el = X_fake_el.reshape((int(len(X_fake_el) / 2), 2))
				X_fake_el = pd.DataFrame(X_fake_el, columns=['x', 'y'])

				plt.plot('x', 'y', c='green', data=X_input_el)
				plt.plot('x', 'y', c='red', data=X_fake_el)
			plt.savefig('plots/gan_e%03d_fake.png' % (i + 1))
			plt.show()
			'''
	return g_model
