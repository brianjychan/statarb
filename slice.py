import pandas as pd
# import numpy as np

kaggle_prices = './kaggle_data/prices.csv'
bloomberg_data = './bloomberg_data.csv'

START_2011_ROW = 118138

# Load Data

# Data format: 
# date,symbol,open,close,low,high,volume
df = pd.read_csv(kaggle_prices,skiprows = 252, header = None)
print(df.head(5))

# get first year
one_year_data = df.iloc[:START_2011_ROW,0:3]
print(one_year_data.iloc[0:4,:])
print(one_year_data.shape[0])

csv = one_year_data.to_csv('year_2010.csv')