import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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

''' PRINCIPAL COMPONENT ANALYSIS '''
pca = PCA()
pca.fit(df.T)
eig_vecs = pca.components_.T
eig_vals = pca.explained_variance_

#print(eig_vecs[:,0].shape)

# CRITERIA 1: THRESHOLD 
def threshold_factors(threshold = 0.55):
    cumulated_variance = np.cumsum(pca.explained_variance_ratio_)
    possible_choices = np.argwhere(cumulated_variance>threshold)
    #i=0
    #while(cumulated_variance[i]< 0.55):
    #    i+=1
    #no_factors = i+1
    print(possible_choices)
    no_factors = possible_choices[0][0]+1
    return no_factors

# CRITERIA 2: BAI AND NG INFORMATION CRITERION
def bai_ng_factors():
    N, T = eig_vecs.shape
    mse = np.zeros(T)
    counting_array = np.array((range(T))) #((N+T)/(N*T))*np.log((N*T)/(N+T))
    penalty = counting_array*(N+T - counting_array)*np.log(N*T)/(N*T)
    for i in range(T):
        loading_mat = eig_vecs[:,:i]
        factors = loading_mat.T@df.values
        mse[i] = mean_squared_error(df.values,loading_mat@factors)
    info_crit = mse + penalty
    #print(info_crit)
    no_factors = np.argmin(info_crit)+1
    return no_factors

# CRITERIA 3: ONATSKI
def onatski_factors(r_max = 15):
    eig_vals_diff = eig_vals[0:(len(eig_vals)-1)]-eig_vals[1:]
    diff = 10
    no_factors = r_max
    j=r_max-1
    while(np.abs(diff)>=1):
        old_no_factors = no_factors
        lambda_values = eig_vals[j:(j+5)]
        xj = (np.array(range(5)))**(2./3)
        reg = LinearRegression().fit(xj.reshape(-1,1),lambda_values)
        beta = reg.coef_ 
        delta = 2*np.abs(beta)
        possible_choices = np.argwhere(eig_vals_diff>=delta)
        if len(possible_choices) == 0: 
            no_factors = 0
        else:
            no_factors = np.max(possible_choices)+1
        if no_factors>= r_max:
            no_factors = r_max
        j = no_factors
        diff = no_factors - old_no_factors
    return no_factors


print(threshold_factors())
print(bai_ng_factors())
print(onatski_factors())    
    