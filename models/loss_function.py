import numpy as np

class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size
