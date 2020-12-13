import sys
sys.path.append("..")

import utils
from utils import config as cn
import analysis as an
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from analysis import countrylist as clist
from analysis import selectorall as sall
from analysis import reader as re
import time
from joblib import dump, load


def declaration(numind=cn.NUM_PREDICTORS):
    indicators=sall.fselection()
    predictors,pre2010=sall.indicators()
    indexs=indicators['Unnamed: 0']
    predictors=np.nan_to_num(predictors)
    pre2010=pre2010.fillna(0)
    npre2010=pre2010.iloc[:,indexs].to_numpy()
    x=predictors[:,:-1]
    y=predictors[:,-1]
    print("Predictors obtained\n")
    return x,y,npre2010

#################################################################################
#gradient boosting#
#################################################################################

def GBmodelTrain(ncorr=cn.NUM_PREDICTORS):
    print("running gradient boosting for all countries")
    x,y,npre2010=declaration(ncorr)
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
    name=input("Write the name of the model")
    fullname=name+'GBM.joblib'
    dump(gbm_model, fullname) 
    if ncorr!=cn.NUM_PREDICTORS:
        return gbm_model
    
def GBmodelPredict(): 
    name=input("Write the name of the model for predicting, without the extension GBM.jlib")
    fullname=name+'GBM.joblib'
    x,y,npre2010=declaration()
    gbm_model=load(fullname)
    results_df = X_test.copy()
    results_df["y_real"] = y_test
    results_df["y_pred"] = gbm_y_pred.astype(float)
    results_df["err"] = results_df["y_real"] - results_df["y_pred"]
    results_df["%_err"] = results_df["err"] / results_df["y_real"] * 100
    prediction=gbm_model.predict(npre2010[:,:-1]).astype(float)
    return prediction


########################################################################################
#multilinear#
#######################################################################################



def multilinearTrain():
    print("running multilinear model for all countries")
    x,y,npre2010=declaration()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    t0 = time.time()
    linear_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t0} seconds")
    name=input("Write the name of the model")
    fullname=name+'ML.joblib'
    dump(linear_model, fullname) 
    
    
    
def multilinearPredict():   
    name=input("Write the name of the model for predicting, without the extension ML.jlib")
    fullname=name+'ML.joblib'
    gbm_model=load(fullname)
    x,y,npre2010=declaration()
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


def GBmodelCorr(ncorr=1329):
    print("running gradient boosting for getting the indicators")
    x,y,npre2010=declaration(ncorr)
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
    name=input("Write the name of the model")
    fullname=name+'GBM.joblib'
    dump(gbm_model, fullname) 
    if ncorr!=cn.NUM_PREDICTORS:
        return gbm_model