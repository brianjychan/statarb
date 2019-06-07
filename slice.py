import pandas as pd
# import numpy as np

kaggle_prices = './kaggle_data/prices.csv'
bloomberg_data = './bloomberg_data.csv'

START_2011_ROW = 118138
# START_2010_ROW = 253
# NEW_START_2011_ROW = 118391
START_2012_ROW = 236578
# original 236831 - 253 + 1

# Load Data

# Data format: 
# date,symbol,open,close,low,high,volume
df = pd.read_csv(kaggle_prices,skiprows = 252, header = None)
print(df.head(5))

# get 2010 year
one_year_data = df.iloc[START_2011_ROW:START_2012_ROW,0:3]
print(one_year_data.iloc[0:4,:])
print(one_year_data.shape[0])

csv = one_year_data.to_csv('year_2011.csv')