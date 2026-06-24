#!/usr/bin/env python3
"""Classical regressor zoo (sklearn + optional xgboost/lightgbm).

Each builder takes an Optuna trial and returns a configured estimator. Model
selection (which name to build) happens in :mod:`plm_regressor.search` via the
size-filtered registry, so these builders take an explicit ``name`` rather than
suggesting it themselves.
"""

from __future__ import annotations

from typing import Tuple

from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor

    HAS_LGB = True
except Exception:  # pragma: no cover
    LGBMRegressor = None
    HAS_LGB = False


def build_classical_model(
    name: str,
    trial: "object",
    random_state: int,
    use_gpu: bool,
    n_features: int,
    n_samples_train: int,
) -> Tuple[str, BaseEstimator]:
    if name == "ridge":
        model = Ridge(alpha=trial.suggest_float("ridge_alpha", 1e-4, 1e3, log=True), random_state=random_state)
    elif name == "sgd":
        model = SGDRegressor(
            loss=trial.suggest_categorical("sgd_loss", ["squared_error", "huber"]),
            penalty=trial.suggest_categorical("sgd_penalty", ["l2", "l1", "elasticnet"]),
            alpha=trial.suggest_float("sgd_alpha", 1e-7, 1e-1, log=True),
            learning_rate="adaptive",
            eta0=trial.suggest_float("sgd_eta0", 1e-4, 1e-1, log=True),
            max_iter=5000,
            early_stopping=True,
            random_state=random_state,
        )
    elif name == "elasticnet":
        model = ElasticNet(
            alpha=trial.suggest_float("enet_alpha", 1e-5, 1e1, log=True),
            l1_ratio=trial.suggest_float("enet_l1_ratio", 0.05, 0.95),
            max_iter=20000,
            random_state=random_state,
        )
    elif name == "huber":
        model = HuberRegressor(
            epsilon=trial.suggest_float("huber_epsilon", 1.05, 2.0),
            alpha=trial.suggest_float("huber_alpha", 1e-6, 1e-1, log=True),
            max_iter=2000,
        )
    elif name == "bayesian_ridge":
        model = BayesianRidge(
            alpha_1=trial.suggest_float("br_alpha_1", 1e-8, 1e-2, log=True),
            alpha_2=trial.suggest_float("br_alpha_2", 1e-8, 1e-2, log=True),
            lambda_1=trial.suggest_float("br_lambda_1", 1e-8, 1e-2, log=True),
            lambda_2=trial.suggest_float("br_lambda_2", 1e-8, 1e-2, log=True),
        )
    elif name == "svr_rbf":
        model = SVR(
            kernel="rbf",
            C=trial.suggest_float("svr_C", 1e-2, 1e3, log=True),
            epsilon=trial.suggest_float("svr_epsilon", 1e-3, 1.0, log=True),
            gamma=trial.suggest_categorical("svr_gamma", ["scale", "auto"]),
        )
    elif name == "knn":
        model = KNeighborsRegressor(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 2, 25),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            p=trial.suggest_int("knn_p", 1, 2),
        )
    elif name == "mlp":
        hidden = trial.suggest_categorical("mlp_hidden", ["64", "128", "256", "128_64", "256_128"])
        hidden_map = {
            "64": (64,),
            "128": (128,),
            "256": (256,),
            "128_64": (128, 64),
            "256_128": (256, 128),
        }
        model = MLPRegressor(
            hidden_layer_sizes=hidden_map[hidden],
            alpha=trial.suggest_float("mlp_alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
            max_iter=3000,
            early_stopping=True,
            random_state=random_state,
        )
    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            n_jobs=1,
            random_state=random_state,
        )
    elif name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("et_n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("et_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", "log2", None]),
            n_jobs=1,
            random_state=random_state,
        )
    elif name == "hist_gb":
        model = HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float("hgb_learning_rate", 1e-3, 0.2, log=True),
            max_depth=trial.suggest_int("hgb_max_depth", 2, 12),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 15, 255),
            l2_regularization=trial.suggest_float("hgb_l2", 1e-8, 1e1, log=True),
            min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 50),
            random_state=random_state,
        )
    elif name == "pls":
        pls_max_components = max(2, min(20, n_features, max(2, n_samples_train - 1)))
        model = PLSRegression(n_components=trial.suggest_int("pls_n_components", 2, pls_max_components))
    elif name == "kernel_ridge":
        model = KernelRidge(
            alpha=trial.suggest_float("kr_alpha", 1e-4, 1e2, log=True),
            kernel=trial.suggest_categorical("kr_kernel", ["rbf", "laplacian", "poly"]),
            gamma=trial.suggest_float("kr_gamma", 1e-6, 1e-1, log=True),
        )
    elif name == "gpr":
        length_scale = trial.suggest_float("gpr_length_scale", 1e-2, 1e2, log=True)
        noise = trial.suggest_float("gpr_noise", 1e-8, 1e0, log=True)
        nu = trial.suggest_categorical("gpr_nu", [0.5, 1.5, 2.5])
        kernel = 1.0 * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise)
        model = GaussianProcessRegressor(kernel=kernel, random_state=random_state, normalize_y=False)
    elif name == "xgboost":
        if not HAS_XGB:
            raise ValueError("xgboost not installed")
        params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.2, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 10),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-6, 10.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 1.0, log=True),
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": 1,
            "tree_method": "hist",
        }
        if use_gpu:
            params["device"] = "cuda"
        model = XGBRegressor(**params)
    elif name == "lightgbm":
        if not HAS_LGB:
            raise ValueError("lightgbm not installed")
        params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 255),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-6, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        if use_gpu:
            params["device"] = "gpu"
        model = LGBMRegressor(**params)
    else:
        raise ValueError(f"Unknown classical model '{name}'")

    return name, model
