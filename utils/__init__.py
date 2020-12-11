import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import sqlite3
import analysis as an
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error