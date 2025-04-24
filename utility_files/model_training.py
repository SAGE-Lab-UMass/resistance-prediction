# model_training.py

import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, roc_auc_score
from utility_files.fused_ridge_gpu import FusedRidge  

def train_ridge_model(X_train, y_train, X_test, y_test):
    ridge = RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10]).fit(X_train, y_train)
    ridge_score = ridge.score(X_test, y_test)
    ridge_auc = roc_auc_score(y_test, ridge.predict(X_test))
    ridge_mse = mean_squared_error(y_test, ridge.predict(X_test))
    return ridge, ridge_score, ridge_auc, ridge_mse

def train_lasso_model(X_train, y_train, X_test, y_test):
    lasso = LassoCV(max_iter=10000, cv=5, random_state=42,
                    alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    lasso.fit(X_train, y_train)
    lasso_score = lasso.score(X_test, y_test)
    lasso_auc = roc_auc_score(y_test, lasso.predict(X_test))
    lasso_mse = mean_squared_error(y_test, lasso.predict(X_test))
    return lasso, lasso_score, lasso_auc, lasso_mse

def train_fused_ridge(X_train, y_train, X_test, y_test, fused_ridge_params):
    fused_ridge = FusedRidge(**fused_ridge_params)
    fused_ridge.fit(X_train, y_train)
    y_pred_test = fused_ridge.predict(X_test)
    test_mse = fused_ridge.calculate_mse(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_test)
    return fused_ridge, test_mse, roc_auc
