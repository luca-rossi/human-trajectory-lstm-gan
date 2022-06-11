import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math

#config
MAX = 20


def plot_sequences(df, df_regr, x_min, x_max, y_min, y_max):
	plt.axis([x_min, x_max, y_min, y_max])
	plt.plot('x', 'y', data=df)
	plt.scatter('x', 'y', s=10, c='black', data=df, label="Sequence points")
	plt.plot(df['x'].iloc[0], df['y'].iloc[0], marker='o', markersize=10, color="green", label="Entry point")
	plt.plot(df['x'].iloc[-1], df['y'].iloc[-1], marker='o', markersize=10, color="yellow", label="Exit point")
	plt.plot(df_regr['x'], df_regr['y'], 'r', label='Fitted line')

	plt.legend()
	plt.show()
	return


def linear_regression(df):
	std_x = df['x'].std()
	std_y = df['y'].std()
	is_vertical = std_y > std_x
	xs, ys = (df['x'], df['y']) if not is_vertical else (df['y'], df['x'])

	slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
	ys = slope * xs + intercept

	error = 0
	for id, row in df.iterrows():
		point = np.array((row['x'], row['y'])) if not is_vertical else np.array((row['y'], row['x']))
		error += ((slope * point[0]) - point[1] + intercept) ** 2
	error = math.sqrt(error / (len(df) * (1 + slope ** 2)))

	return (xs if not is_vertical else ys), (ys if not is_vertical else xs), error


def describe_dataset(dataset, max=MAX, verbose=0):
	print(dataset.shape)
	print(dataset.head(max))
	print(dataset.describe())
	if verbose > 0:
		std_errs = []
		tot_avg_err = 0
		x_min = dataset['x'].min()
		x_max = dataset['x'].max()
		y_min = dataset['y'].min()
		y_max = dataset['y'].max()
		id_count = 0
		for id, df in dataset.groupby('id'):
			xs, ys, std_err = linear_regression(df)
			df_regr = pd.DataFrame({
				'x': xs,
				'y': ys,
			})
			std_errs.append(std_err)
			tot_avg_err += std_err
			if verbose > 1 and id_count < max:
				print('Std err: ' + str(std_err))
				plot_sequences(df, df_regr, x_min, x_max, y_min, y_max)
				id_count += 1
		tot_avg_err /= len(std_errs)

		tot_std_err = 0
		for std_err in std_errs:
			tot_std_err += (std_err - tot_avg_err) ** 2
		tot_std_err = math.sqrt(tot_std_err / len(std_errs))

		print('Tot avg err: ' + str(tot_avg_err))
		print('Tot std err: ' + str(tot_std_err))
