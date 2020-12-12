import sys
sys.path.append("..")

import utils
from utils import config as cn
from utils import io
import analysis as an
import sqlite3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

import time


def declaration():
    numind=cn.NUM_PREDICTORS
    indicators=io.selection(numind)
    predictors,pre2010=io.indicators(numind)
    indexs=indicators.index.values.tolist()
    predictors=np.nan_to_num(predictors)
    pre2010=pre2010.fillna(0)
    npre2010=pre2010[indexs].to_numpy()
    x=predictors[:,:-1]
    y=predictors[:,-1]
    return x,y,npre2010



def GBmodel():
    print("running gradient boosting for all countries")
    x,y,npre2010=declaration()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    gbm_hyperparams = {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'loss': 'ls'
    }
    gbm_model = GradientBoostingRegressor(**gbm_hyperparams)
    t0 = time.time()
    gbm_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t0} seconds")
    t0 = time.time()
    gbm_y_pred = gbm_model.predict(X_test)
    print(f"Elapsed time predicting: {time.time() - t0} seconds")
    print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, gbm_y_pred)}")

    results_df = X_test.copy()
    results_df["y_real"] = y_test
    results_df["y_pred"] = gbm_y_pred.astype(float)
    results_df["err"] = results_df["y_real"] - results_df["y_pred"]
    results_df["%_err"] = results_df["err"] / results_df["y_real"] * 100
    prediction=gbm_model.predict(npre2010[:,:-1]).astype(float)
    return prediction

def GBmodel_train():
    print("running gradient boosting for all countries")
    x,y,npre2010=declaration()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    gbm_hyperparams = {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'loss': 'ls'
    }
    gbm_model = GradientBoostingRegressor(**gbm_hyperparams)
    t0 = time.time()
    gbm_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t0} seconds")
    return gbm_model

def GBmodel_test():
    t0 = time.time()
    gbm_y_pred = gbm_model.predict(X_test)
    print(f"Elapsed time predicting: {time.time() - t0} seconds")
    print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, gbm_y_pred)}")

    results_df = X_test.copy()
    results_df["y_real"] = y_test
    results_df["y_pred"] = gbm_y_pred.astype(float)
    results_df["err"] = results_df["y_real"] - results_df["y_pred"]
    results_df["%_err"] = results_df["err"] / results_df["y_real"] * 100
    prediction=gbm_model.predict(npre2010[:,:-1]).astype(float)
    return prediction

def multilinearmodel():
    print("running gradient boosting for all countries")
    x,y,npre2010=declaration()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    t0 = time.time()
    linear_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t0} seconds")
    t0 = time.time()
    gbm_y_pred = gbm_model.predict(X_test)
    print(f"Elapsed time predicting: {time.time() - t0} seconds")
    print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, gbm_y_pred)}")

    results_df = X_test.copy()
    results_df["y_real"] = y_test
    results_df["y_pred"] = linear_model.astype(float)
    results_df["err"] = results_df["y_real"] - results_df["y_pred"]
    results_df["%_err"] = results_df["err"] / results_df["y_real"] * 100
    prediction=linear_model.predict(npre2010[:,:-1]).astype(float)
    return prediction
