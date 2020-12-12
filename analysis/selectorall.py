import sys
sys.path.append("..")
import utilsGDP
from utilsGDP import config as cn
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