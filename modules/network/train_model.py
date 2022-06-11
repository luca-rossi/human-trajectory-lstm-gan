from keras.callbacks import EarlyStopping, ModelCheckpoint
from config import VARS


def train_model(dataset, model, model_path='model.h5', n_features=8, batch_size=1,
                n_epochs=100, patience=10, val_train_ratio=0.2, shuffle=False):
	'''
	fit a network to training data
	'''
	n_vars = len(VARS)
	tot_features = n_features * n_vars
	X, y = dataset[:, 0:tot_features], dataset[:, tot_features:]

	callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
				ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)]
	# fit network
	model.fit(X, y, validation_split=val_train_ratio, epochs=n_epochs, batch_size=batch_size, verbose=2, shuffle=shuffle, callbacks=callbacks)
	return model
