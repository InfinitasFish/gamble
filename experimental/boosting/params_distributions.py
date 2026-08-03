

xgb_param_distr = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 10.0],
}


lgbm_param_distr = {
    "n_estimators": [100, 200, 300],
    "num_leaves": [7, 15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [-1, 5, 8, 12],
    "min_child_samples": [5, 20, 40],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 10.0],
}


et_param_distr = {
    "n_estimators": [100, 300],
    "max_features": ["sqrt", "log2", 0.3, 0.7],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "bootstrap": [True, False],
}
