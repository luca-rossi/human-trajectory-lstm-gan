import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from config import VARS


scalers = []


def load_scalers():
	global scalers
	for i in range(len(VARS)):
		scalers.append(joblib.load('scaler' + str(i) + '.pkl'))


def save_scalers():
	global scalers
	for i in range(len(VARS)):
		joblib.dump(scalers[i], 'scaler' + str(i) + '.pkl')


def normalize_dataset(dataset):
	global scalers
	for i in range(len(VARS)):
		values = dataset[VARS[i]].values.reshape(len(dataset[VARS[i]]), 1)
		if i < len(scalers):
			dataset[VARS[i]] = scalers[i].transform(values)
		else:
			scaler = MinMaxScaler(feature_range=(-1, 1))    #RobustScaler(quantile_range=(25, 75))
			dataset[VARS[i]] = scaler.fit_transform(values)
			scalers.append(scaler)
	return dataset


def denormalize_dataset(results):
	global scalers
	inverted = list()
	for i in range(len(results)):
		# create array from results
		result = np.array(results[i])
		inv_scale = [None] * len(result)

		# invert results for each var
		n_vars = len(VARS)
		for i in range(n_vars):
			result_var = result[i::n_vars]
			result_var = result_var.reshape(1, len(result_var))
			inv_scale_var = scalers[i].inverse_transform(result_var)
			inv_scale_var = inv_scale_var[0, :]
			inv_scale[i::n_vars] = inv_scale_var

		# store
		inverted.append(inv_scale)
	return inverted
