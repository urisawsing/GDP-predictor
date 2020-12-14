import sys
sys.path.append("..")
import utils
from utils import config as cn
from analysis import reader as re
import pandas as pd
import numpy as np
from analysis import countrylist as clist
from models import model_countries_together as allc

def fselection(answer=False):
    print("Selecting the best indicators...\n")
    data=re.readall()
    num=clist.numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized = data.pivot_table(index = ['CountryCode','Year'], columns = "IndicatorCode", values = 'Value')
    resized['COUNTRYENC']=resized.index.get_level_values(0)
    resized['NextYearGDP']=resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    print("Do you want to use the selection of indicators predefined?")
    de=input("In the case you want type Yes, if not type No")
    if de=="No"or de=="no" or de=="N" or de=="noup":
        print("This option may take a while(like 10min), please wait")
        model=allc.GBmodelTrain(ncorr=1329)
        a=model.feature_importances_
        df=pd.DataFrame(a)
        high=df[0].nlargest(50)
        high.to_csv("bestindicators.csv")
    else:
        high=pd.read_csv("bestindicators.csv")
        
    print("Indicators selected\n")
    return high,data,num

def selection(answer=False):
    print("Getting the best indicators...")
    data=re.readall()
    num=clist.numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized = data.pivot_table(index = ['CountryCode','Year'], columns = "IndicatorCode", values = 'Value')
    high=pd.read_csv("bestindicators.csv")
    print("Done\n")
    return high




def indicators(chosen,data,num):
    print("Obtaining the data...\n")
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized = data.pivot_table(index = ['CountryCode','Year'], columns = "IndicatorCode", values = 'Value')
    resized['COUNTRYENC']=resized.index.get_level_values(0)
    resized['NextYearGDP']=resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    indicators=pd.DataFrame(chosen)
    indexs=indicators['Unnamed: 0']
    indexs=indexs.to_numpy()
    indexs=list(indexs)
    bestcorr=resized.iloc[:,indexs]  
    #resized=resized.drop(index=2010,level=1)
    final=bestcorr.to_numpy()
    print("Data obtained\n")
    return resized