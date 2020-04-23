import warnings
import numpy as np
from keras.callbacks import Callback


class GetBest(Callback):
	"""Get the best model at the end of training.

	WRITTEN BY LOUIS YANG (louis925): https://github.com/keras-team/keras/issues/2768

	# Arguments
		monitor: quantity to monitor.
		verbose: verbosity mode, 0 or 1.
		mode: one of {auto, min, max}.
			The decision
			to overwrite the current stored weights is made
			based on either the maximization or the
			minimization of the monitored quantity. For `val_acc`,
			this should be `max`, for `val_loss` this should
			be `min`, etc. In `auto` mode, the direction is
			automatically inferred from the name of the monitored quantity.
		period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
				 callbacks=callbacks)
	"""

	def __init__(self, monitor='val_loss', verbose=0,
				 mode='auto', period=1):
		super(GetBest, self).__init__()
		self.monitor = monitor
		self.verbose = verbose
		self.period = period
		self.best_epochs = 0
		self.epochs_since_last_save = 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('GetBest mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.monitor_op = np.greater
				self.best = -np.Inf
			else:
				self.monitor_op = np.less
				self.best = np.Inf

	def on_train_begin(self, logs=None):
		self.best_weights = self.model.get_weights()

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			# filepath = self.filepath.format(epoch=epoch + 1, **logs)
			current = logs.get(self.monitor)
			if current is None:
				warnings.warn('Can pick best model only with %s available, '
							  'skipping.' % (self.monitor), RuntimeWarning)
			else:
				if self.monitor_op(current, self.best):
					if self.verbose > 0:
						print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
							  ' storing weights.'
							  % (epoch + 1, self.monitor, self.best,
								 current))
					self.best = current
					self.best_epochs = epoch + 1
					self.best_weights = self.model.get_weights()
				else:
					if self.verbose > 0:
						print('\nEpoch %05d: %s did not improve' %
							  (epoch + 1, self.monitor))

	def on_train_end(self, logs=None):
		if self.verbose > 0:
			print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
													   self.best))
		self.model.set_weights(self.best_weights)
