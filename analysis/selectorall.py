import sys
sys.path.append("..")
import utils
from utils import config as cn
from analysis import reader as re
import pandas as pd
import numpy as np
from analysis import countrylist as clist

def selection(numcorr):
    data=re.readall()
    num=clist.numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized= data.Value.groupby([data.CountryCode,data.Year, data.IndicatorCode]).sum().unstack().fillna(0).astype(float)
    resized['COUNTRYENC']=resized.index.get_level_values(0)
    resized['NextYearGDP']=resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    resized=resized.drop(index=2010,level=1)
    corr=resized.corr()
    gdpcorr=corr['NextYearGDP']
    absgdp=abs(gdpcorr)
    high=absgdp.nlargest(numcorr)
    return high

def indicators(numcorr):
    data=re.readall()
    num=clist.numadapt()
    cat=data['CountryCode'].astype('category')
    data['CountryCode']=cat
    correction=data['CountryCode'].cat.codes
    data['CountryCode']=correction
    resized= data.Value.groupby([data.CountryCode,data.Year, data.IndicatorCode]).sum().unstack().fillna(0).astype(float)
    resized['COUNTRYENC']=resized.index.get_level_values(0)
    resized['NextYearGDP']=resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)
    swapped=resized.swaplevel(0,1)
    df2010=swapped.loc[[2010]]
    resized=resized.drop(index=2010,level=1)
    return resized, df2010