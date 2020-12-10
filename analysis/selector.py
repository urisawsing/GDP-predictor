import sys
sys.path.append("..")
import utils
from utils import config as cn
from analysis import reader as re
import pandas as pd
import numpy as np

def selection(country,numcorr):
    data=re.read(country)
    resized= data.Value.groupby([data.Year, data.IndicatorCode]).sum().unstack().fillna(0).astype(float)
    corr=resized.corr()
    gdpcorr=corr['NY.GDP.MKTP.KD.ZG']
    absgdp=abs(gdpcorr)
    high=absgdp.nlargest(numcorr)
    return high

def indicators(country,numcorr):
    data=re.read(country)
    resized= data.Value.groupby([data.Year, data.IndicatorCode]).sum().unstack().fillna(0).astype(float)
    return resized