import numpy as np

class ReLu:

    @staticmethod
    def activate(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)


class Sigmoid:

    @staticmethod
    def activate(x):
        x = np.clip(x, -5, 5)
        z = np.where(x >= 0, np.exp(-x), np.exp(x))
        return np.where(x >= 0, 1 / (1 + z), z / (1 + z))

    @staticmethod
    def derivative(x):
        sig = Sigmoid.activate(x)

        return sig * (1 - sig)