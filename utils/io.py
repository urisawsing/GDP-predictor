"""Reads the database."""
import sqlite3

import pandas as pd

from .config import DATABASE_PATH


# readers and writers
def readall():
    """Reads the full database.

    Returns
    -------
    pandas.DataFrame with the whole data
    """
    print("Reading the SQL file...\n")
    db = sqlite3.connect(DATABASE_PATH)
    r=pd.read_sql_query("SELECT * FROM CountryIndicators", db)
    print("Done\n")
    return r

def ind_names():
    """
    Returns
    -------
    pandas.DataFrame with the indicators
    
    """
    db = sqlite3.connect(DATABASE_PATH)
    return pd.read_sql_query("SELECT * FROM Indicators", db)



def writeall(dataset):
    """
    Parameters
    ----------
    dataset: pandas.DataFrame
        the predictions from the model
        
    Summary
    -------
    Writes the predictions obtained into the SQL
    """ 
    print("Writing on the SQL file...\n")
    connection = sqlite3.connect(DATABASE_PATH)
    db = connection.cursor()
    line=[]
    dataset=dataset.to_numpy()
    for i in range(len(dataset)):
        line.append(dataset[i,0])
        line.append("2010")
        line.append(str(dataset[i,1]))
        db.execute('''INSERT INTO EstimatedGPDGrowth
                    ( CountryCode , Year , Value ) 
                    VALUES(?,?,?)''',line)
        connection.commit()
        line=[]
    print("SQL written")
    db.close()
    
    
    
    
    
# countrylist
def countryarray():
    """

    Returns
    -------
    pandas.DataFrame
        the list of countries with the name and the 
        countrycode
    """
    db = sqlite3.connect(DATABASE_PATH)
    dl = pd.read_sql_query("SELECT * from Countries", db)
    return dl.iloc[:, 0:2]


def numadapt():
    """
    Same as `countryarray` but with an extra column.

    Returns
    -------
    pandas.DataFrame
        the list of countries with the name, countrycode and index
    """
    print("Getting list of countries...\n")
    countries = countryarray()
    countries['num'] = list(range(1, len(countries) + 1))
    print("Done\n")
    return countries


# selectors
def _append_category_to_countrycode(df):
    cat = df['CountryCode'].astype('category')
    df['CountryCode'] = cat
    df['CountryCode'] = df['CountryCode'].cat.codes




def selection(answer=False):
    """

    Parameters
    ----------
    answer: boolean

    Returns
    -------
    high: pandas.DataFrame
        The best indicators from a csv file
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
        Indicators selected for the model
        that will train the model
        
    data: pandas.DataFrame
        whole data non preprocessed
        extracted from the SQL
        
    num: pandas.DataFrame
        Country list with the countrycodes and
        the numbers

    Returns
    -------
        resized: pandas.DataFrame
            preprocessed dataframe in where the rows are the year and country 
            and the columns is each indicator, just needs one last step selecting the
            useful columns
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
