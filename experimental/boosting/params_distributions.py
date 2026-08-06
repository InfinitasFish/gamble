from scipy import stats


xgb_param_distr = {
    "n_estimators": stats.randint(100, 300),
    "max_depth": stats.randint(3, 10),
    "learning_rate": stats.uniform(0.001, 0.2),
    "subsample": stats.uniform(0.5, 1.0),
    "reg_alpha": stats.uniform(0.01, 1.0),
    "reg_lambda": stats.uniform(0.0, 10.0),
}


lgbm_param_distr = {
    "n_estimators": stats.randint(100, 300),
    "num_leaves": stats.randint(7, 67),
    "learning_rate": stats.uniform(0.001, 0.2),
    "max_depth": stats.randint(1, 15),
    "subsample": stats.uniform(0.5, 1.0),
    "reg_alpha": stats.uniform(0.01, 1.0),
    "reg_lambda": stats.uniform(0.0, 10.0),
}


et_param_distr = {
    "n_estimators": stats.randint(100, 200, 300),
    "max_depth": stats.randint(1, 10),
    "min_samples_split": stats.randint(2, 10),
    "min_samples_leaf": stats.randint(1, 10),
}
