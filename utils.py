import math
from numpy.random import randn


def get_values(dataset):
	return dataset.values[:, 1:]


def get_dataset_path(dataset, retail=False, prep=False):
	return 'data/' + ('retail' if retail else 'official') + '/' + dataset + '_dataset' + ('_prep' if prep else '') + '.txt'


def lcm(a, b):
	return int(abs(a * b) / math.gcd(int(a), int(b)) if a and b else 0)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input
