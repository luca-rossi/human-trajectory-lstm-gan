'''
from loading to test

Official:
	- No interpolation needed
	- Time as integer
Retail:
	- Interpolation needed
	- Time as datetime

Every script outputs the whole dataset, each sequence has a different id, other scripts only have to groupby
Ungroupby with filter
Real preprocessing steps
Describe with different levels of verbose
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.animation import writers

from config import *
from modules.dataset.filter_dataset import filter_dataset
from modules.dataset.filter_sequences import filter_sequences
from modules.dataset.get_sequences import get_sequences
from modules.dataset.load_dataset import load_dataset
from modules.dataset.sample_sequences import sample_sequences
from modules.dataset.save_dataset import save_dataset
from modules.dataset_prep.get_supervised_dataset import get_supervised_dataset
from modules.dataset_prep.normalize_dataset import normalize_dataset, denormalize_dataset
from modules.dataset_prep.split_train_test import split_train_test
from modules.network.evaluate import evaluate_model
from modules.network.test_model import test_model
from modules.network.train_model import train_model
from utils import get_values, get_dataset_path, lcm

writers.reset_available_writers()

#plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

def load(dataset_name):
	retail = dataset_name in DATASETS_RETAIL
	is_prep = dataset_name not in DATASETS_FIRST_LOAD
	path = get_dataset_path(dataset_name, retail=retail, prep=is_prep)
	if is_prep:
		return load_dataset(path=path, has_timestamp=retail, sort=True)
	dataset = load_dataset(path=path, delimiter=DATASET_DELIMITER, has_timestamp=retail, has_description=retail, sort=True, id='tag_id')
	if retail:
		#dataset = filter_dataset(dataset, x_min=-100, x_max=100, y_min=-100, y_max=100, start_date='2019-08-10 10:00:00', end_date='2019-08-10 10:30:00')
		dataset = get_sequences(dataset, max_stop_time=2000)
		dataset = filter_sequences(dataset, min_seq_len=12, min_seq_std=0.1)        # pre-filter for fast sampling
		dataset = sample_sequences(dataset, sampling_period='2000ms')
	dataset = filter_sequences(dataset, min_seq_len=12, min_seq_std=0.1)            # filter again for reduced sequences
	save_dataset(dataset, get_dataset_path(dataset_name, retail=retail, prep=True))
	return dataset


def get_datasets():
	datasets_train = list()
	dataset_test = None

	for i in range(0, len(DATASETS_TRAIN)):
		# load the dataset
		dataset_name = DATASETS_TRAIN[i]
		dataset_train = load(dataset_name)

		# TODO
		dataset_train['x'] -= dataset_train['x'].mean()
		dataset_train['y'] -= dataset_train['y'].mean()


		# if the test set is one of the training sets, split it
		if dataset_name is DATASET_TEST:
			dataset_train, dataset_test = split_train_test(dataset_train, train_test_ratio=TRAIN_TEST_RATIO)
		# increment the ids of each dataset so they don't get mixed
		if i > 0:
			dataset_train['id'] += datasets_train[i - 1]['id'].max() + 1
		# append the loaded dataset to the datasets list
		datasets_train.append(dataset_train)

	# merge the training sets
	dataset_train = pd.concat(datasets_train).reset_index(drop=True)

	# load the test set if it's not been loaded before as one of the (split) training sets...
	if dataset_test is None:
		dataset_test = load(DATASET_TEST)

		# TODO
		dataset_test['x'] -= dataset_test['x'].mean()
		dataset_test['y'] -= dataset_test['y'].mean()

	return dataset_train, dataset_test


def preprocess(dataset):
	if all(elem in DATASETS_RETAIL for elem in DATASETS_TRAIN):
		dataset = filter_dataset(dataset, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, start_date=START_DATE, end_date=END_DATE)
	dataset = filter_sequences(dataset, min_seq_len=MIN_SEQ_LEN, min_seq_std=MIN_SEQ_STD)
	dataset = normalize_dataset(dataset)
	dataset = get_supervised_dataset(dataset, seq_len=SEQ_LEN, n_features=N_FEATURES, reapply_filter=REAPPLY_FILTER, augment=AUGMENT)
	return dataset


def train_data(dataset):
	residual = len(dataset) % lcm(BATCH_SIZE, (1 / VAL_TRAIN_RATIO))
	if residual > 0:
		dataset = dataset.iloc[:-residual]
	print('Train shape: ' + str(len(dataset)))
	dataset = get_values(dataset)
	model = train_model(dataset, load_model=LOAD_MODEL, model_path=MODEL_PATH, model_type=MODEL_TYPE,
						n_features=N_FEATURES, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, n_neurons=N_NEURONS,
						patience=PATIENCE, dropout=DROPOUT, val_train_ratio=VAL_TRAIN_RATIO, shuffle=SHUFFLE)
	return model


def test_and_evaluate_data(dataset, model):
	residual = len(dataset) % BATCH_SIZE
	if residual > 0:
		dataset = dataset.iloc[:-residual]
	print('Test shape: ' + str(len(dataset)))
	dataset = get_values(dataset)
	predictions, ground_truth = test_model(dataset, model, model_type=MODEL_TYPE, n_features=N_FEATURES, batch_size=BATCH_SIZE)
	predictions = denormalize_dataset(predictions)
	ground_truth = denormalize_dataset(ground_truth)

	ade, fde, n_ade, n_fde = evaluate_model(predictions, ground_truth)
	print('ADE: ' + str(ade))
	print('FDE: ' + str(fde))
	print('Normalized ADE: ' + str(n_ade))
	print('Normalized FDE: ' + str(n_fde))
	return predictions, ground_truth

#def update(i):


def plot_results(dataset, predictions, ground_truth):
	# TODO
	dt = 0.05
	n_vars = len(VARS)
	dataset = get_values(dataset)
	for j in range(0, 2):
		fig, ax = plt.subplots()
		fig.set_tight_layout(True)
		ax.axis([-1, 1, -1, 1])
		# Plot a scatter that persists (isn't redrawn) and the initial line.


		row = dataset[j]
		pred = np.array(predictions[j])
		row = row.reshape((int(len(row) / n_vars), n_vars))
		row = pd.DataFrame(row, columns=VARS)
		pred = pred.reshape((int(len(pred) / n_vars), n_vars))
		pred = pd.DataFrame(pred, columns=VARS)

		pred = normalize_dataset(pred)

		data = row[:N_FEATURES]
		ground_truth = row[N_FEATURES:]

		ax.scatter('x', 'y', c='blue', s=1, data=data)
		ax.scatter('x', 'y', c='green', s=1, data=ground_truth)
		ax.scatter('x', 'y', c='red', s=1, data=pred)

		print('--- Data')
		print(data)
		print('. GT')
		print(ground_truth)
		print('. PD')
		print(pred)

		# plt.scatter('x', 'y', c='black', s=1, data=data)
		# plt.scatter('x', 'y', c='black', s=1, data=ground_truth)
		# plt.scatter('x', 'y', c='black', s=1, data=pred)
		# data_a = data.values.tolist()
		print('----- Data Values:')
		print((data.iloc[0]['x'], data.iloc[0]['y']))
		print((data.iloc[1]['x'], data.iloc[1]['y']))
		line, = ax.plot([], [], 'b', lw=2)
		time_template = 'time = %.1fs'
		#time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

		thisx = []
		thisy = []

		def init():
			line.set_data([], [])
			#time_text.set_text('')
			return line#, time_text

		def update(i):
			thisx.append(data.iloc[i]['x'])
			thisy.append(data.iloc[i]['y'])

			line.set_data(thisx, thisy)
			#time_text.set_text(time_template % (i * dt))
			return line#, time_text

		anim = FuncAnimation(fig, update, frames=np.arange(0, len(data)), interval=200, init_func=init)

		# plt.plot('x', 'y', c='blue', data=data)
		# plt.plot('x', 'y', c='green', data=ground_truth)
		# plt.plot('x', 'y', c='red', data=pred)


		anim.save('line.gif', dpi=80, writer='pillow')


		# plt.show()


# load the datasets and separate them in a training set and a test set
dataset_train, dataset_test = get_datasets()

# prepare scaler        TODO
ds = pd.concat([dataset_train, dataset_test])
ds = normalize_dataset(ds)

#describe_dataset(dataset_train, verbose=2)
#describe_dataset(dataset_test, verbose=2)

# preprocess the training and test sets
dataset_train = preprocess(dataset_train)
dataset_test = preprocess(dataset_test)

# fit and test
model = train_data(dataset_train)
predictions, ground_truth = test_and_evaluate_data(dataset_test, model)

# plot_results
plot_results(dataset_test, predictions, ground_truth)



# TODO fix scaler (LSTM test set may not be in [-1, 1])
# TODO custom loss
# TODO plot train/val
# TODO NV_ADE, NV_FDE
# TODO refactor results[i % N_VAR]
# TODO cross validation
# TODO space / time matrix
# TODO GAN (generate N samples and choose the one with less distance from all the others?)
