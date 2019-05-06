import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

kaggle_prices = './kaggle_data/prices.csv'
bloomberg_data = './bloomberg_data.csv'
year_2010 = './year_2010.csv'


'''Load Data and Convert to Matrix form, Drop NA entries'''

df = pd.read_csv(year_2010)
df = df.drop('index', axis=1)
#print(df.head(5))

#print(len(df.symbol.unique()))

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

df = df.pivot(index = 'symbol', columns='date', values = 'price')

#csv = df.to_csv('first_matrix.csv')
#print("--------")
#print(df.head(5))
df = df.dropna()
#print(df.shape)

''' Data normalization '''
tomorrow_price = df.iloc[:,1:]
today_price = df.iloc[:,0:(df.shape[1]-1)]
df = pd.DataFrame((tomorrow_price.values-today_price.values)/today_price.values,columns=today_price.columns, index = today_price.index)
df_mean = df.mean(axis=1)
df_std = df.std(axis=1) 
df = df.sub(df_mean,axis=0)
df = df.div(df_std,axis=0)
#print(df.mean(axis=1))
#print(df.std(axis=1))
#



''' Dimension Reduction '''
#cov_mat = df.T.cov()
cov_mat = (1./(df.shape[1]-1))*df.dot(df.T)
#print(cov_mat.head(5))
#print(cov_mat.head(5))

eig_val, eig_vec = np.linalg.eigh(cov_mat)
eig_pair = []
for i in range(len(eig_val)):
    eig_pair.append((eig_val[i],eig_vec[:,i]))
eig_pair.sort(key=lambda x: x[0], reverse=True)


#print(eig_val.shape,eig_vec.shape)
#print(eig_val, eig_vec)
#plt.hist(eig_val)
#plt.savefig('hist_eigvals')
#print(sum(eig_val))

threshold = 0.55
sum=0
i=0
while(sum < threshold*df.shape[0]):
    eig = eig_pair[i]
    sum = sum+eig[0]
    i+=1
no_factors = i 
    
#transform into a new subspace
weight_mat = np.empty((no_factors,df.shape[0]))
for i in range(no_factors):
    weight_mat[i,:] = eig_pair[i][1].reshape(df.shape[0],)
print(weight_mat.shape)

projected_data = weight_mat@df.values
print(projected_data.shape)

loading_mat = weight_mat.T

