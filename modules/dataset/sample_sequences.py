def sample(df, sampling_period):
	df = df.resample(sampling_period, on='time').mean().interpolate()
	df['time'] = df.index
	return df


def sample_sequences(dataset, sampling_period='2000ms'):
	'''
	sample the sequences in the dataset
	'''
	return dataset.groupby('id').apply(lambda group: sample(group, sampling_period=sampling_period)).reset_index(drop=True)
