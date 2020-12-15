import os
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .io import indicators
from .io import readall
from .io import numadapt
from .io import writeall
from .io import _append_category_to_countrycode
from .config import NUM_PREDICTORS, MODELS_PATH, EXHAUSTIVE_ITER



#################################################################################
# gradient boosting#
#################################################################################


def GBmodelTrain(ncorr=NUM_PREDICTORS):
    """
    Parameters
    ----------
    ncorr: int
        the number of predictors to use
    Returns
    -------
    gbm_model: GradientBoostingRegressor or None
        only returns the model if the number of predictors is different from the default one.
    Summary
    -------
    Computes the gradient boosting for the best 50 indicators, saves the model in the models file and the rsquared in the logs
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
    fullname = os.path.join(MODELS_PATH, name + 'GBM.joblib')
    dump(gbm_model, fullname)

    return gbm_model if ncorr != NUM_PREDICTORS else None

    return r2_score(y_test, gbm_y_pred)


def GBmodelPredict():
    """
    Returns
    -------
    prediction: 
    array of the prediction for 2010.
    
    Summary
    -------
    Loads the model you write and predicts the 2011 gdp, the results can be saved on the SQL database 
    also allows to save the predictions in a csv with the same name of the model
    """
    name = input("Write the name of the model for predicting, without the extension GBM.jlib")
    fullname = name + 'GBM.joblib'
    fullname = os.path.join(MODELS_PATH, name + 'GBM.joblib')
    x, y, npre2010 = declaration()
    gbm_model = load(fullname)
    
    prediction = gbm_model.predict(npre2010).astype(float)
    print(prediction)
    print("Some postprocessing...\n")
    num=numadapt()
    pdf=pd.DataFrame(prediction)
    numc=num["CountryCode"]
    pred=pd.DataFrame(numc)
    pred["Predicted GDPGrowth"]=pdf
    writeall(pred)
    print("do you want to save the predictions in a csv?\n")
    a=input("In case you want type Y\n")
    if a=="Y":
        predname=os.path.join(MODELS_PATH,"predGBM"+name+".csv")
        np.savetxt(predname, prediction)
        print("Prediction saved")
        
    return prediction


########################################################################################
# multilinear #
#######################################################################################


def multilinearTrain():
    """
    Returns 
    -------
    r2_score(y_test, linear_y_pred): float
        Rsquared of the linear model
    Summary
    -------
    Computes the multilinear training for the best 50 indicators, saves the model in the models file and the rsquared in the logs
    
    """
    print("running multilinear model for all countries")

    x, y, npre2010 = declaration()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    t0 = time.time()
    linear_model.fit(X_train, y_train)
    linear_y_pred = linear_model.predict(X_test)
    print(f"Elapsed time training: {time.time() - t0} seconds")
    print(f"RMSE: {mean_squared_error(y_test, linear_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, linear_y_pred)}")
    print(f"Elapsed time training: {time.time() - t0} seconds")

    name = input("Write the name of the model")
    fullname = os.path.join(MODELS_PATH, name + 'ML.joblib')
    dump(linear_model, fullname)
    return r2_score(y_test, linear_y_pred)


def multilinearPredict():
    """
    Returns
    -------
    prediction: float
        the prediction for 2011 GDP.
        
    Summary
    -------
    Loads the model you write and predicts the GDP of 2011 with a multilinear model, 
    the results can be saved on the SQL database ,also allows to save the predictions
    in a csv with the same name of the model
    """
    
    
    name = input("Write the name of the model for predicting, without the extension ML.jlib")
    fullname = name + 'ML.joblib'
    linear_model = load(fullname)
    x, y, npre2010 = declaration()
    t0 = time.time()
    prediction = linear_model.predict(npre2010).astype(float)
    
    print("do you want to save the predictions in a csv?\n")
    a=input("In case you want type Y\n")
    if a=="Y":
        np.savetxt("predictiondataML.csv", prediction)
        print("Prediction saved")
    return prediction


############################################################################################################
## Other Functions##
##########################################################################################################

def declaration(numind=NUM_PREDICTORS):
    """
    Parameters
    ----------
    numind: int
        the number of predictors to use
    Returns
    -------
    x: pandas.DataFrame
        Set of predictors written on the proper way that the model wil use
        to predict the y.
        
    y: pandas.DataFrame
        Array of GDP growths or the variable the model will try to predict
        
    npre2010: pandas.DataFrame
        Dataset or predictors for the 2010 which will be used
        to predict the GDP growth
        
    Summary
    -------
    Grouping of all the preprocessing functions before running the predictions, 
    reads and prepares the data in order the  model just has to be trained
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


def GBmodelCorr(ncorr=1329):
    """
    Parameters
    ----------
    ncorr: int
        the number of predictors to use
    Returns
    -------
    gbm_model: GradientBoostingRegressor or None
        only returns the model if the number of predictors is different from the default one.
    
    Summary
    -------
    This function is reserved to compute the best indicators from the whole set, it is just a 
    Gradient Boosting which the interesting part is not the model but the importances
    
    """
    print("running gradient boosting for getting the indicators")

    x, y, npre2010 = declaration(ncorr)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    gbm_hyperparams =  {
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
    print(f"Elapsed time training: {time.time() - t0} seconds")

    t0 = time.time()
    gbm_y_pred = gbm_model.predict(X_test)
    print(f"Elapsed time predicting: {time.time() - t0} seconds")

    print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, gbm_y_pred)}")

    name = input("Write the name of the model")
    fullname = name + 'GBM_1329.joblib'
    dump(gbm_model, fullname)

    return gbm_model if ncorr != NUM_PREDICTORS else None


def fselection():
    """??
    Returns
    -------
    high: pandas.DataFrame
        Set of best indicators with the index
    data: pandas.DataFrame
        Non preprocessed data from the SQL
    num: pandas.DataFrame
        Relation between countrycode and index
        
    Summary
    -------
    First selection, equal to selection but will ask if you want to reselect the indicators
    instead of the predefined ones
    """
    print("Selecting the best indicators...\n")
    data = readall()
    num = numadapt()

    # append the category to the code
    _append_category_to_countrycode(data)
    print("Do you want to use the selection of indicators predefined?")
    ans = input("In the case you want type Yes, if not type No")
    if ans in ['No', 'no', 'N', 'noup']:
        print("This option may take a while(like 10min), please wait")
        model = GBmodelTrain(ncorr=1329)
        a = model.feature_importances_
        df = pd.DataFrame(a)
        high = df[0].nlargest(50)
        high.to_csv("bestindicators.csv")
    else:
        high = pd.read_csv("bestindicators.csv")

    print("Indicators selected\n")
    return high, data, num