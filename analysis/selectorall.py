import sys
sys.path.append("..")
import utils
from utils import config as cn
from analysis import reader as re
import pandas as pd
import numpy as np
from analysis import countrylist as clist
from models import model_countries_together as allc

def selection():
    data=re.readall()
    num=clist.numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized = data.pivot_table(index = ['CountryCode','Year'], columns = "IndicatorCode", values = 'Value')
    print("Do you want to use the selection of indicators predefined?")
    de=input("In the case you want type Yes, if not type No")
    if de=="Yes"or de=="yes" or de=="Y":
        print("The")
        model=allc.GBmodelTrain(ncorr=1329)
        a=model.feature_importances_
        df=pd.DataFrame(a)
        high=df[0].nlargest(50)
        high.to_csv("bestindicators.csv")
    else:
        high=pd.read_csv("bestindicators.csv")
    return high




def indicators(ncorr):
    data=re.readall()
    num=clist.numadapt()
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