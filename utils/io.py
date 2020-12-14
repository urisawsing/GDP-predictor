"""Reads the database."""
import sqlite3

import pandas as pd

from .models import GBmodelTrain
from .config import DATABASE_PATH


# readers
def readall():
    """Reads the full database.

    Returns
    -------
    pandas.DataFrame
    """
    db = sqlite3.connect(DATABASE_PATH)
    return pd.read_sql_query("SELECT * FROM CountryIndicators", db)


def read(country):
    """Selects from the database only the passed country.

    Parameters
    ----------
    country: str
        the country whose data will be returned.

    Returns
    -------
    pandas.DataFrame
    """
    db = sqlite3.connect(DATABASE_PATH)
    di = pd.read_sql_query("SELECT * FROM CountryIndicators", db)
    return di[di['CountryCode'].str.contains(country)]


# countrylist
def countryarray():
    """??

    Returns
    -------
    pandas.DataFrame
        the data
    """
    db = sqlite3.connect(DATABASE_PATH)
    dl = pd.read_sql_query("SELECT * from Countries", db)
    return dl.iloc[:, 0:2]


def numadapt():
    """Same as `countryarray` but with an extra column.

    Returns
    -------
    pandas.DataFrame
        the data
    """
    countries = countryarray()
    countries['num'] = list(range(1, len(countries) + 1))
    return countries


# selectors
def _append_category_to_countrycode(df):
    cat = df['CountryCode'].astype('category')
    df['CountryCode'] = cat
    df['CountryCode'] = df['CountryCode'].cat.codes


def fselection(answer=False):
    """??

    Parameters
    ----------
    answer: boolean

    Returns
    -------
    high: pandas.DataFrame
        ?
    data: pandas.DataFrame
        ?
    num: pandas.DataFrame
        ?

    """
    print("Selecting the best indicators...\n")
    data = readall()
    num = numadapt()

    # append the category to the code
    _append_category_to_countrycode(data)

    # resize the data
    # resized = data.pivot_table(index=['CountryCode', 'Year'], columns="IndicatorCode", values='Value')
    # resized['COUNTRYENC'] = resized.index.get_level_values(0)
    # resized['NextYearGDP'] = resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)

    print("Do you want to use the selection of indicators predefined?")
    ans = input("In the case you want type Yes, if not type No")
    if ans in ['No', 'no', 'N', 'noup']:
        print("This option may take a while(like 10min), please wait")
        model = GBmodelTrain(ncorr=1329)
        a = model.feature_importances_
        df = pd.DataFrame(a)
        high = df[0].nlargest(50)
        high.to_csv("bestindicators.csv")
    else:
        high = pd.read_csv("bestindicators.csv")

    print("Indicators selected\n")
    return high, data, num


def selection(answer=False):
    """??

    Parameters
    ----------
    answer: boolean

    Returns
    -------
    high: pandas.DataFrame
        ?
    """
    print("Getting the best indicators...")
    data = readall()
    _append_category_to_countrycode(data)
    # resized = data.pivot_table(index=['CountryCode', 'Year'], columns="IndicatorCode", values='Value')
    high = pd.read_csv("bestindicators.csv")
    print("Done\n")
    return high


def indicators(chosen, data, num):
    """??

    Parameters
    ----------
    chosen: pandas.DataFrame
        ?
    data: pandas.DataFrame
        ?
    num: pandas.DataFrame
        ?

    Returns
    -------
    resized: pandas.DataFrame
        ?
    """
    print("Obtaining the data...\n")
    _append_category_to_countrycode(data)

    resized = data.pivot_table(index=['CountryCode', 'Year'], columns="IndicatorCode", values='Value')
    resized['COUNTRYENC'] = resized.index.get_level_values(0)
    resized['NextYearGDP'] = resized['NY.GDP.MKTP.KD.ZG'].shift(periods=-1)

    indicators = pd.DataFrame(chosen)
    indexs = indicators['Unnamed: 0']
    indexs = indexs.to_numpy()
    indexs = list(indexs)
    # bestcorr = resized.iloc[:, indexs]
    # resized=resized.drop(index=2010,level=1)
    # final = bestcorr.to_numpy()
    print("Data obtained\n")
    return resized
