import sys
sys.path.append("..")

import utils
from utils import config as cn
import sqlite3
import pandas as pd
import countrylist as cl

def numadapt():
    l=cl.countryarray()
    
    