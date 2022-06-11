import pandas as pd


def get_subgroups_ids(df, max_stop_time):
	return ((df['id'] != df['id'].shift()) | (df['time'] > (df['time'].shift() + pd.to_timedelta(max_stop_time, unit='ms')))).cumsum()


def get_sequences(dataset, max_stop_time=2000):
	'''
	split the dataset in multiple sequences of different length sorted by time
	'''
	dataset = dataset.sort_values(by=['id', 'time'])        #resort just in case
	dataset['id'] = get_subgroups_ids(dataset, max_stop_time)
	return dataset
