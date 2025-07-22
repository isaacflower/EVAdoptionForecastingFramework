import gpflow
import tensorflow as tf
import numpy as np
from scipy.interpolate import make_smoothing_spline
from ev_adoption_forecasting_package.transforms import probit

class Spline():
    def __init__(self, X_train, Y_train) -> None:
        self.X_train = X_train
        self.Y_train = Y_train
        self.spline = None

    def fit_spline(self):
        self.spline = make_smoothing_spline(self.X_train, self.Y_train, lam=None)

    def evaluate(self, X):
        if self.spline is None:
            raise ValueError("Spline not fitted. Please call fit_spline() first.")
        logistic_value = self.spline(X)
        return logistic_value

class CustomMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, mean_function):
        super().__init__()
        self.mean_function = mean_function

    def __call__(self, X):
        X = tf.cast(X, dtype=tf.float64)
        X_np = X.numpy()
        mean_values = self.mean_function.evaluate(X_np)
        probit_mean_values = probit(mean_values)
        return tf.convert_to_tensor(probit_mean_values, dtype=tf.float64)