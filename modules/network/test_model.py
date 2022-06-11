from config import VARS
from utils import generate_latent_points


def test_model(dataset, model, n_features=8, batch_size=1, is_gan=False, latent_dim=16):
	'''
	evaluate the persistence model
	'''
	n_vars = len(VARS)
	tot_features = n_features * n_vars
	predictions = list()
	ground_truth = list()
	for i in range(0, len(dataset), batch_size):
		X, y = dataset[i:i+batch_size, 0:tot_features], dataset[i:i+batch_size, tot_features:]
		prediction = None
		if is_gan:
			x_gan = generate_latent_points(latent_dim, batch_size)
			prediction = model.predict([X, x_gan])
		else:
			prediction = model.predict(X, batch_size=batch_size)
		for row in prediction:
			predictions.append(row)
		for row in y:
			ground_truth.append(row)
	return predictions, ground_truth
