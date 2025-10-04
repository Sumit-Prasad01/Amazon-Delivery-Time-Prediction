from scipy.stats import randint, uniform

LIGHTGBM_PARAMS = {
    "n_estimators": randint(200, 1001),  # Discrete uniform range for number of trees
    "learning_rate": uniform(0.01, 0.19), # Continuous uniform range (0.01 to 0.20)
    "max_depth": [-1, 5, 10, 20],        # Retained as a list for specific fixed values
    "num_leaves": randint(31, 201),      # Discrete uniform range for number of leaves
    "subsample": uniform(0.6, 0.4),      # Continuous uniform range (0.6 to 1.0)
    "colsample_bytree": uniform(0.6, 0.4), # Continuous uniform range (0.6 to 1.0)
    "reg_alpha": uniform(0, 1),          # Continuous uniform range (0 to 1)
    "reg_lambda": uniform(0, 1)           # Continuous uniform range (0 to 1)
}



RANDOM_SEARCH_PARAMS = {
    'n_iter' : 30,              # number of random combinations to try
    'scoring' : "neg_root_mean_squared_error",
    'cv' : 3,                   # 3-fold cross-validation
    'verbose' : 2,
    'random_state' : 42,
    'n_jobs' : -1
}