import numpy as np

def accuracy(y_true, y_pred):
    return float(np.mean((np.argmax(y_pred, axis=-1) == y_true).astype('float32')))

def mse(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))
