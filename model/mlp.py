import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from sklearn.neural_network import MLPRegressor

from constants import RANDOM_STATE, MAX_ITER


def init_mlp_uni_reg(max_iter: int=MAX_ITER, random_state: int=RANDOM_STATE) -> MLPRegressor:
    mlp_reg = MLPRegressor(loss="squared_error", hidden_layer_sizes=(150,), activation='relu', learning_rate_init=0.001,
                           shuffle=False, verbose=True, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    return mlp_reg


def train_mlp_uni_reg(mlp_reg: MLPRegressor, X: np.ndarray, y: np.ndarray) -> MLPRegressor:
    mlp_reg.fit(X, y)
    return mlp_reg
