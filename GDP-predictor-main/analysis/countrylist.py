import sys
sys.path.append("..")

import utils
from utils import config as cn
import sqlite3
import pandas as pd

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
