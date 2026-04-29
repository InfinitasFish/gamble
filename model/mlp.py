from constants import RANDOM_STATE, MAX_ITER

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


def init_mlp_uni_reg(max_iter: int=MAX_ITER, random_state: int=RANDOM_STATE) -> MLPRegressor:
    mlp_reg = MLPRegressor()
    pass


if __name__ == "__main__":
    print(59)
