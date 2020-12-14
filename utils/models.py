import os
import time

import numpy as np
from joblib import dump, load
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .io import fselection, indicators
from .config import NUM_PREDICTORS, MODELS_PATH, GBM_HYP


def declaration(numind=NUM_PREDICTORS):
    """???

    Parameters
    ----------
    numind: int
        the number of predictors to use

    Returns
    -------
    x: pandas.DataFrame
        ?
    y: pandas.DataFrame
        ?
    npre2010: pandas.DataFrame
        ?
    """
    high, data, num = fselection()
    predictors = indicators(high, data, num)
    predictors = predictors.fillna(0)

    indexs = high['Unnamed: 0']
    indexs = indexs.tolist()

    names = predictors.columns.tolist()
    high["Unnamed: 0"] = high["Unnamed: 0"] + 1
    definitivo = [names[ind] for ind in indexs]
    definitivo.append('COUNTRYENC')

    swapped = predictors.swaplevel(0, 1)
    df2010 = swapped.loc[[2010]]
    pre2010 = df2010.fillna(0)
    npre2010 = pre2010[definitivo]

    x = predictors[definitivo]
    x = x.fillna(0)
    y = predictors['NextYearGDP']
    print("Predictors obtained\n")

    return x, y, npre2010

#################################################################################
# gradient boosting#
#################################################################################


def GBmodelTrain(ncorr=NUM_PREDICTORS):
    """???

    Parameters
    ----------
    ncorr: int
        the number of predictors to use

    Returns
    -------
    gbm_model: GradientBoostingRegressor or None
        only returns the model if the number of predictors is different from the default one.
    """
    print("running gradient boosting for all countries...")
    t0 = time.time()
    x, y, npre2010 = declaration(ncorr)
    print(f"Elapsed time preprocessing: {time.time() - t0} seconds")

    t1 = time.time()
    print("Training the model...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    gbm_hyperparams = {
        'n_estimators': 2000,
        'max_depth': 5,
        'learning_rate': 0.1,
        'loss': 'huber',
        'criterion': 'mse',
        'validation_fraction': 0.01,
        'n_iter_no_change': 20
    }
    gbm_model = GradientBoostingRegressor(**gbm_hyperparams)
    t0 = time.time()
    gbm_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t1} seconds")

    t1 = time.time()
    gbm_y_pred = gbm_model.predict(X_test)
    print(f"Elapsed time predicting: {time.time() - t1} seconds")

    print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, gbm_y_pred)}")
    print(f"Total execution time: {time.time() - t0} seconds")

    name = input("Write the name of the model")
    name1 = name + 'GBM.joblib'
    dirname = os.path.dirname(name1)
    fullname = os.path.join(MODELS_PATH, dirname)
    dump(gbm_model, fullname)

    return gbm_model if ncorr != NUM_PREDICTORS else None


def GBmodelPredict():
    """???

    Returns
    -------
    prediction: float
        the prediction for 2010.
    """
    name = input("Write the name of the model for predicting, without the extension GBM.jlib")
    fullname = name + 'GBM.joblib'

    x, y, npre2010 = declaration()
    gbm_model = load(fullname)
    print(npre2010)
    prediction = gbm_model.predict(npre2010).astype(float)
    print(prediction)
    np.savetxt("primeraprediccio.csv", prediction)
    return prediction


########################################################################################
# multilinear #
#######################################################################################


def multilinearTrain():
    """???"""
    print("running multilinear model for all countries")

    x, y, npre2010 = declaration()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    t0 = time.time()
    linear_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t0} seconds")

    name = input("Write the name of the model")
    fullname = name + 'ML.joblib'
    dump(linear_model, fullname)


def multilinearPredict():
    """???

    Returns
    -------
    prediction: float
        the prediction for 2010.
    """
    name = input("Write the name of the model for predicting, without the extension ML.jlib")
    fullname = name + 'ML.joblib'
    gbm_model = load(fullname)
    x, y, npre2010 = declaration()
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
    prediction = linear_model.predict(npre2010[:, :-1]).astype(float)
    return prediction

##################################################################################################


def GBmodelCorr(ncorr=1329):
    """???

    Parameters
    ----------
    ncorr: int
        the number of predictors to use

    Returns
    -------
    gbm_model: GradientBoostingRegressor or None
        only returns the model if the number of predictors is different from the default one.
    """
    print("running gradient boosting for getting the indicators")

    x, y, npre2010 = declaration(ncorr)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    gbm_hyperparams = GBM_HYP
    gbm_model = GradientBoostingRegressor(**gbm_hyperparams)
    t0 = time.time()
    gbm_model.fit(X_train, y_train)
    print(f"Elapsed time training: {time.time() - t0} seconds")

    t0 = time.time()
    gbm_y_pred = gbm_model.predict(X_test)
    print(f"Elapsed time predicting: {time.time() - t0} seconds")

    print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, gbm_y_pred)}")

    name = input("Write the name of the model")
    fullname = name + 'GBM.joblib'
    dump(gbm_model, fullname)

    return gbm_model if ncorr != NUM_PREDICTORS else None
