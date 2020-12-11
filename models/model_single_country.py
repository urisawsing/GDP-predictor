import sys
sys.path.append("..")

import utils
from utils import config as cn
import analysis as an
import sqlite3
import pandas as pd

def declaration(name,indicators,predictors):
    #declaration of targets in the correct format
    target=['NextYearGDP']
    target=target[0]
    #declaration of predictors in the correct format and creation of "next year GDP"
    gdp=pd.DataFrame(predictors['NY.GDP.MKTP.KD.ZG'])
    gdps=gdp['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    gdps=pd.DataFrame(gdps)
    gdps['NY.GDP.MKTP.KD.ZG']=gdps['NY.GDP.MKTP.KD.ZG'].fillna(0)
    predictors['NextYearGDP']=gdps
    indicators=pd.DataFrame(indicators)
    indexs=indicators.index.values.tolist()
    x=predictors[indexs]
    y=predictors[target]
    return x,y


def multilinear(name,indicators,predictors,tablebool):
    print("running multilinear predictor for the country",name,'\n')
    x,y=declaration(name,indicators,predictors)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_y_pred = linear_model.predict(X_test)
    results_df = X_test.copy()
    results_df["y_real"] = y_test
    results_df["y_pred"] = linear_y_pred.astype(int)
    results_df["err"] = results_df["y_real"] - results_df["y_pred"]
    results_df["%_err"] = results_df["err"] / results_df["y_real"] * 100
    print(f"RMSE: {mean_squared_error(y_test, linear_y_pred)**0.5}")
    print(f"R^2: {r2_score(y_test, linear_y_pred)}")
    predict=linear_model.predict(x)
    predictTable=x.copy()
    predictTable['Prediction']=predict
    if tablebool==True:
        return predict, prediction10, predictTable
    elif tablebool==False:
        return predict, prediction10
    
    
    
        

def gradboost(name,indicators,predictors,tablebool):
    print("running gradient boost predictor for the country",name,'\n') 
    x,y=declaration(name,indicators,predictors)
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
    prediction10=results_df.at[2010,'y_pred']
    predict=gbm_model.predict(x)
    prediction=results_df.at[2010,'y_pred']
    predictTable=x.copy()
    predictTable['Prediction']=predict
    if tablebool==True:
        return predict, prediction10, predictTable
    elif tablebool==False:
        return predict, prediction10
    