import sys
sys.path.append("..")
import utils
from utils import config as cn
from analysis import reader as re
import pandas as pd
import numpy as np
from analysis import countrylist as clist
from models import singlecountry as scn

def runallcountries():
    tablebool=False
    countries=countryarray()
    numind=config.NUM_PREDICTORS
    indicators=se.selection(name,numind)
    predictors=se.indicators(name,numind)
    for country in countries:
        indicators=se.selection(country[1],numind)
        predictors=se.indicators(country[1],numind)
        a,b=scn.multilinear(country[1],indicators,predictors,tablebool)
