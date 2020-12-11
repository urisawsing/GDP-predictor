import sys
sys.path.append("..")

import utils
from utils import config as cn
import analysis as an
import sqlite3
import pandas as pd

#readers
def readall():
    db=sqlite3.connect(cn.DATABASE_PATH)
    cur = db.cursor()
    di = pd.read_sql_query("SELECT * from CountryIndicators", db)
    return di
 
def read(country):
    db=sqlite3.connect(cn.DATABASE_PATH)
    cur = db.cursor()
    di = pd.read_sql_query("SELECT * from CountryIndicators", db)
    dfcountry = di[di['CountryCode'].str.contains(country)]
    di=0
    return dfcountry

#countrylist 
def countryarray():
    db=sqlite3.connect(cn.DATABASE_PATH)
    cur = db.cursor()
    dl = pd.read_sql_query("SELECT * from Countries", db)
    countries=dl.iloc[:,0:2]
    return countries

def numadapt():
    countries= countryarray()
    num=[]
    for i in range(len(countries)):
    	num.append(i+1)
    countries['num']=num
    return countries

#selectors
def selection(numcorr):
    data=readall()
    num=numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized = data.pivot_table(index = ['CountryCode','Year'], columns = "IndicatorCode", values = 'Value')
    resized['COUNTRYENC']=resized.index.get_level_values(0)
    resized['NextYearGDP']=resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    resized=resized.drop(index=2010,level=1)
    corr=resized.corr()
    gdpcorr=corr['NextYearGDP']
    absgdp=abs(gdpcorr)
    high=absgdp.nlargest(numcorr)
    return high

def indicators(ncorr):
    data=readall()
    num=numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized = data.pivot_table(index = ['CountryCode','Year'], columns = "IndicatorCode", values = 'Value')
    resized['COUNTRYENC']=resized.index.get_level_values(0)
    resized['NextYearGDP']=resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    chosen=selection(ncorr)
    indicators=pd.DataFrame(chosen)
    indexs=indicators.index.values.tolist()
    bestcorr=resized[indexs]
    swapped=resized.swaplevel(0,1)
    df2010=swapped.loc[[2010]]
    resized=resized.drop(index=2010,level=1)
    final=bestcorr.to_numpy()
    return final, df2010



#def retrieve_training_dataset():
 #   pass


#def retrieve_predict_dataset():
 #   pass
