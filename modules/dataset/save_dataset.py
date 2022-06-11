def save_dataset(df, path, delimiter=',', append = False):
	mode = 'a' if append else 'w'

	'''
	datetime = pd.to_datetime(df['time']).dt
	time = 10 * (datetime.second + \
			60 * (datetime.minute + \
			60 * (datetime.hour + \
			24 * (datetime.day))))
	df = pd.DataFrame({
		'time': time,
		'id': df['id'],
		'x': df['x'],
		'y': df['y'],
	})
	'''
	df.to_csv(path, header = not append, index=False, sep=delimiter, mode=mode)
