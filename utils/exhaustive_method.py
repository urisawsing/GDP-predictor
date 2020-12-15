import os
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from .io import  indicators
from .io import readall
from .io import numadapt
from .io import _append_category_to_countrycode
from .config import NUM_PREDICTORS, MODELS_PATH, EXHAUSTIVE_ITER
from .models import declaration




######################################################################
#EXHAUSTIVE METHODS#
#######################################################################


def ExhaustiveGBM(iterations=EXHAUSTIVE_ITER,ncorr=NUM_PREDICTORS):
    """
    Parameters
    ----------
    iterations: int
        number of iterations the model will train a 
        Gradient Boosting Model
    ncorr: int
        number of indicators used to train the model
    
    Returns
    -------
    R: float
        Highest Rsquared value from all the executions
    
    Summary
    -------
    This function runs the GBM n times and chooses the model which
    achieves a better rsquared, saves the model and returns the rsquared
    
    """
    print("You entered the Exhaustive option for the Gradient boost model\n")
    print("This method will iterate the predictive model", iterations,"times\n")
    
    
    rsq=[]
    mod=[]
    print("running the model for all countries...")
    
    
    t0 = time.time()
    x, y, npre2010 =declaration(ncorr)
    print(f"Elapsed time preprocessing: {time.time() - t0} seconds")
    
    
    print("Entering the training part...")
    for i in range(iterations):
        
        print("Iteration ",i+1," of ",iterations,)
        
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
        t1 = time.time()
        gbm_model.fit(X_train, y_train)
        print(f"Elapsed time training: {time.time() - t1} seconds")

        t1 = time.time()
        gbm_y_pred = gbm_model.predict(X_test)
        print(f"Elapsed time predicting: {time.time() - t1} seconds")
        print(f"RMSE: {mean_squared_error(y_test, gbm_y_pred)**0.5}")
        print(f"R^2: {r2_score(y_test, gbm_y_pred)}")
        mod.append(gbm_model)
        rsq.append(r2_score(y_test, gbm_y_pred))
    
    R=max(rsq)
    rin=rsq.index(R)
    print("R^2 max obtained is",R)
    Exh_model=rsq[rin]
    
    name = input("Write the name of the model")
    fullname = os.path.join(MODELS_PATH, name+'E_GBM.joblib')
    dump(gbm_model, fullname)
        
    
    print(f"Total execution time: {time.time() - t0} seconds")
    return R    







def ExhaustiveML(iterations=EXHAUSTIVE_ITER,ncorr=NUM_PREDICTORS):
    """
    Parameters
    ----------
    iterations: int
        number of iterations the model will train a 
        MultiLinear Model
    ncorr: int
        number of indicators used to train the model
    
    Returns
    -------
    R: float
        Highest Rsquared value from all the executions
    
    Summary
    -------
    This function runs the GBM n times and chooses the model which
    achieves a better rsquared, saves the model and returns the rsquared
    
    
    """
    print("You entered the Exhaustive option for the MultiLinear model\n")
    print("This method will iterate the predictive model", iterations,"times\n")
    
    
    rsq=[]
    mod=[]
    print("running the model for all countries...")
    
    
    t0 = time.time()
    x, y, npre2010 = declaration(ncorr)
    print(f"Elapsed time preprocessing: {time.time() - t0} seconds")
    
    
    print("Entering the training part...")
    for i in range(iterations):
        
        print("Iteration ",i+1," of ",iterations,)
        
        t1 = time.time()
        print("Training the model...")
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42*i)
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        linear_y_pred = linear_model.predict(X_test)
        t1 = time.time()
        print(f"RMSE: {mean_squared_error(y_test, linear_y_pred)**0.5}")
        print(f"R^2: {r2_score(y_test, linear_y_pred)}")
        print(f"Elapsed time training: {time.time() - t0} seconds")
        mod.append(linear_model)
        rsq.append(r2_score(y_test, linear_y_pred))
        linear_model=0
        
    
    R=max(rsq)
    ri=rsq.index(R)
    print("R^2 max obtained is",R)
    Exh_model=mod[ri]
    
    name = input("Write the name of the model")
    fullname = os.path.join(MODELS_PATH, name+'E_ML.joblib')
    dump(Exh_model, fullname)
        
    
    print(f"Total execution time: {time.time() - t0} seconds")
    return R 






