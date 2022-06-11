import pandas as pd
from config import VARS
from config import *



EPS = 0.01


# select multiple non-overlapping sequences
def split_normal(group, seq_len, reapply_filter):
	drop_id = -1
	group['seq_id'] = group['id'].copy()
	for i in range(group.index[0], group.index[0] + len(group), seq_len):
		if reapply_filter and all(group.loc[i:i+seq_len-1, var].std() <= EPS for var in VARS):
			group.loc[i:i + seq_len - 1, 'id'] = drop_id
		else:
			group.loc[i:i+seq_len-1, 'id'] = i
	# remove sequences with low variance
	group = group.loc[group['id'] != drop_id]
	# remove last group with sequence length < seq_len
	residual = len(group) % seq_len
	if residual > 0:
		group = group.iloc[:-residual]
	return group


# augment (and add ID column for when it will be split in train-test)
def split_augment(group, seq_len, reapply_filter):
	group['seq_id'] = group['id'].copy()
	augmented_group = []
	for i in range(group.index[0], group.index[0] + len(group) - seq_len + 1):
		if not reapply_filter or any(group.loc[i:i+seq_len-1, var].std() > EPS for var in VARS):
			rows = group.loc[i:i+seq_len-1].copy()
			rows['id'] = i
			augmented_group.append(rows)
	return pd.concat(augmented_group)


def get_supervised_dataset(df, seq_len=12, n_features=8, reapply_filter=True, augment=False):
	# series to supervised: dataframe with a row for each sequence
	col_names = ['idx']
	# input sequence (t-n, ... t-1)
	for i in range(n_features, 0, -1):
		col_names += ['%s(t-%d)' % (var, i) for var in VARS]
	# forecast sequence (t, t+1, ... t+n)
	col_names += ['%s(t)' % var for var in VARS]
	for i in range(1, seq_len - n_features):
		col_names += ['%s(t+%d)' % (var, i) for var in VARS]

	groups = df.groupby('id')
	if augment:
		groups = groups.apply(lambda group: split_augment(group, seq_len, reapply_filter))
	else:
		groups = groups.apply(lambda group: split_normal(group, seq_len, reapply_filter))
	groups['idx'] = groups['id'].copy()

	# id represents a sequence of seq_len length, seq_id represents the starting sequence,
	# seq_id will become the id in the new dataset and multiple sequences can have the same id,
	# so later we can avoid to use the same sequences in train and test sets in case of data augmentation
	sequences = pd.DataFrame(columns=col_names)
	for (_, seq_id), group in groups.groupby(['idx', 'seq_id']):
		seq_df = pd.DataFrame([[seq_id, *group[VARS].values.flatten()]], columns=col_names)
		sequences = sequences.append(seq_df, ignore_index=True)
	return sequences
