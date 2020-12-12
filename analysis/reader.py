import sys
sys.path.append("..")

import utilsGDP
from utilsGDP import config as cn
import sqlite3
import pandas as pd


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
