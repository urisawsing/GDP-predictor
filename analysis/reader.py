import sys
sys.path.append("..")

import utils
from utils import config as cn
import sqlite3
import pandas as pd


def readall():
    print("Reading the SQL file...")
    db=sqlite3.connect(cn.DATABASE_PATH)
    cur = db.cursor()
    di = pd.read_sql_query("SELECT * from CountryIndicators", db)
    print("File read\n")
    return di
    
def read(country):
    db=sqlite3.connect(cn.DATABASE_PATH)
    cur = db.cursor()
    di = pd.read_sql_query("SELECT * from CountryIndicators", db)
    dfcountry = di[di['CountryCode'].str.contains(country)]
    di=0
    return dfcountry
