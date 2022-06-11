def filter_dataset(dataset, x_min = None, x_max = None, y_min = None, y_max = None, start_date = None, end_date = None):
	'''
	filter the dataset by time and area, can be used at any stage after loading and before preprocessing
	'''
	dataset = dataset.loc[(dataset['x'] >= x_min) &
			            (dataset['x'] <= x_max) &
			            (dataset['y'] >= y_min) &
			            (dataset['y'] <= y_max) &
			            ((start_date is None) or (dataset['time'] > start_date)) &
			            ((end_date is None) or (dataset['time'] <= end_date))]
	return dataset
