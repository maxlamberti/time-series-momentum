from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class MultiLayerPerceptron(BaseEstimator):

    def __init__(self, num_features, hidden_layers=1, hidden_neurons=5, hidden_dropout=0.2, input_dropout=0,
                 hidden_activation='relu', optimizer='adam', kernel_initializer='normal',
                 bias_initializer='normal', output_activation='linear', output_len=1, loss='mse',
                 normalize_features=False):

        # sanitize input
        if not isinstance(hidden_neurons, list):
            hidden_neurons = [int(hidden_neurons) for _ in range(int(hidden_layers))]

        # define and build model
        self.model_template = self._build_model(num_features, input_dropout, hidden_neurons,
                                                hidden_activation, hidden_dropout, kernel_initializer,
                                                bias_initializer, output_len, output_activation)

        # save data structures
        self.loss = loss
        self.optimizer = optimizer
        self.fit_history = []
        self.normalize_features = normalize_features
        self.scaler = StandardScaler()
        self.model = deepcopy(self.model_template)
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=256, shuffle=True, verbose=0, callbacks=[]):
        """Fit the MLP model to training data."""

        if self.normalize_features:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        have_val_data = (X_val is not None) and (y_val is not None)
        val_data = (X_val, y_val) if have_val_data else None

        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=val_data,
            shuffle=shuffle,
            callbacks=callbacks
        )
        self.fit_history.append(history)

    def predict(self, X):
        """Make predictions using the MLP model."""

        if self.normalize_features:
            X = self.scaler.transform(X)

        preds = self.model.predict(X)

        return preds

    def reset(self):
        """Reset the MLP model weights. Erases training history."""
        self.fit_history = []
        self.model = deepcopy(self.model_template)
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def summary(self):
        """Print a summary of the model."""
        self.model.summary()

    def save(self, filepath):
        """Save the model to an hdf5 file."""
        self.model.save(filepath)

    @staticmethod
    def _build_model(num_features, input_dropout, hidden_neurons, hidden_activation, hidden_dropout,
               kernel_initializer, bias_initializer, output_len, output_activation):
        """Construct a generic MLP model."""

        model = Sequential()

        # visible layer dropout
        model.add(Dropout(input_dropout, input_shape=(int(num_features),)))

        # hidden layers
        for num_neurons in hidden_neurons:
            model.add(Dense(int(num_neurons), activation=hidden_activation, kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer))
            if hidden_dropout > 0:
                model.add(Dropout(hidden_dropout))

        # output layer
        model.add(Dense(output_len, activation=output_activation, kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer))

        return model


if __name__ == '__main__':

    import numpy as np
    from sklearn.metrics import mean_squared_error

    # generate fake data
    num_train = 9000
    num_test = 1000
    num_data = num_train + num_test
    X = np.random.uniform(0, 1, (num_data, 3))
    y = X[:, 0] ** 2 + X[:, 1] - X[:, 2] ** 1.5 + np.random.normal(0, 0.01, num_data)

    # test MLP model
    MLP = MultiLayerPerceptron(X.shape[1], hidden_neurons=[6, 5, 4, 3], normalize_features=True)
    MLP.fit(X[:num_test, :], y[:num_test], epochs=10, verbose=1)
    MLP.reset()
    MLP.fit(X[:num_test, :], y[:num_test], epochs=10, verbose=1)
    preds = MLP.predict(X[num_test:, :])
    MLP.summary()
    print("MSE: {}".format(mean_squared_error(y[num_test:], preds)))
