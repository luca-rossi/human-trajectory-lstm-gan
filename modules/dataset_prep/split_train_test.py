import pandas as pd
import random


def split_train_test(dataset, train_test_ratio=0.7):
	# put same id sequences in the same set
	groups = [dataset for _, dataset in dataset.groupby('id')]
	random.shuffle(groups)
	split_pos = int(len(groups) * train_test_ratio)
	return pd.concat(groups[:split_pos]).reset_index(drop=True), pd.concat(groups[split_pos:]).reset_index(drop=True)
