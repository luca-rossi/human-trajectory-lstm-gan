import pandas as pd
import numpy as np
import math
from scipy import stats
from config import VARS

K_LENGTH = 14
K_LIN = 0.8
NF_CEILING = 10

def compute_timestep_error(predictions, ground_truth, traj_lengths, traj_lins, timestep):
	n_vars = len(VARS)
	first_column = timestep * n_vars
	last_column = (timestep + 1) * n_vars - 1
	predictions = predictions.loc[:, first_column:last_column]
	ground_truth = ground_truth.loc[:, first_column:last_column]

	diff = predictions - ground_truth
	diff['diff'] = sum([diff.iloc[:, i] ** 2 for i in range(n_vars)])
	diff['diff'] = diff['diff'].apply(lambda x: math.sqrt(x))
	nfs = (K_LENGTH / traj_lengths) * (K_LIN / traj_lins)
	nfs = nfs.apply(lambda nf: math.sqrt(nf))#min(math.sqrt(nf), NF_CEILING))
	diff['normal_diff'] = diff['diff'] * nfs

	diff = diff.replace([np.inf, -np.inf], np.nan)
	# TODO group by 20 and make all equal to the best (NOT THE SAME THING but who cares)
	#error = diff['diff'].rolling(10).min()
	#normal_error = diff['normal_diff'].rolling(10).min()
	#error = error.mean(skipna=True)
	#normal_error = normal_error.median(skipna=True)

	error = diff['diff'].mean(skipna=True)
	normal_error = diff['normal_diff'].median(skipna=True)

	print('t+%d\t\tADE: %f\tN-ADE: %f' % ((timestep + 1), error, normal_error))
	return error, normal_error


def compute_traj_lengths(dataset, n_coordinates):
	n_vars = len(VARS)
	n_diffs = (n_coordinates - 1) * n_vars
	squares = pd.DataFrame([(dataset.iloc[:, i+n_vars] - dataset.iloc[:, i]) ** 2 for i in range(n_diffs)])
	lengths = squares.apply(lambda x: sum([math.sqrt(sum(x[i:i+n_vars])) for i in range(0, n_diffs, n_vars)]))
	return lengths

def compute_traj_lins(dataset):
	n_vars = len(VARS)
	lins = []
	for i in range(0, len(dataset)):
		row = np.array(dataset.iloc[i])
		row = row.reshape((int(len(row) / n_vars), n_vars))
		row = pd.DataFrame(row, columns=VARS)

		std_x = row['x'].std()
		std_y = row['y'].std()
		is_vertical = std_y > std_x
		xs, ys = (row['x'], row['y']) if not is_vertical else (row['y'], row['x'])

		slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)

		error = 0
		for id, row in row.iterrows():
			point = np.array((row['x'], row['y'])) if not is_vertical else np.array((row['y'], row['x']))
			error += ((slope * point[0]) - point[1] + intercept) ** 2
		error = math.sqrt(error / (len(row) * (1 + slope ** 2)))
		lins.append(error)

	return pd.Series(lins)

def evaluate_model(predictions, ground_truth):
	n_coordinates = int(len(ground_truth[0]) / len(VARS))
	predictions = pd.DataFrame(predictions)
	ground_truth = pd.DataFrame(ground_truth)

	# TODO drop from ground_truth and predictions traj_length=0 index

	traj_lengths = compute_traj_lengths(ground_truth, n_coordinates)
	traj_lins = compute_traj_lins(ground_truth)

	# nf = (K_LENGTH / traj_lengths) * (K_LIN / traj_lins)

	errors, normal_errors = zip(*[compute_timestep_error(predictions, ground_truth, traj_lengths, traj_lins, i) for i in range(n_coordinates)])

	ade = sum(errors) / n_coordinates
	fde = errors[-1]
	n_ade = sum(normal_errors) / n_coordinates
	n_fde = normal_errors[-1]
	#tue = ((errors[1] / errors[0]) ** (len(errors) - 1)) / (errors[-1] / errors[0])
	tue = (errors[1] - errors[0]) * (len(errors) - 1) / (errors[-1] - errors[0])
	# (N - 1) / ((2 - 1) * N)

	return ade, fde, n_ade, n_fde, tue
