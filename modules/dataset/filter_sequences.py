from config import VARS


def filter(df, min_seq_len, min_seq_std):
	return (len(df) >= min_seq_len) and all(df[var].std() > min_seq_std for var in VARS)


def filter_sequences(dataset, min_seq_len=12, min_seq_std=0.1):
	'''
	filter and split the dataset in multiple sequences of different length
	'''
	return dataset.groupby('id').filter(lambda group: filter(group, min_seq_len, min_seq_std)).reset_index(drop=True)
