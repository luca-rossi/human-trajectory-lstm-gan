import pandas as pd


CART = 'Cart'


def load_dataset(path, delimiter=',', has_timestamp=True, has_description=False,
                 id='id', time='time', x='x', y='y', description='description', sort=True):
	'''
	load the dataset, rename columns and sort by time
	'''

	df = pd.read_csv(path, delimiter = delimiter)

	if has_description:
		df = df[[id, time, x, y, description]]
		df = df.loc[(df[description] == CART)]
	else:
		df = df[[id, time, x, y]]

	df = pd.DataFrame({
		'id': df[id],
		'time': pd.to_datetime(df[time]) if has_timestamp else df[time],
		'x': df[x],
		'y': df[y],
	})
	return df.sort_values(by=['id', 'time']) if sort else df
