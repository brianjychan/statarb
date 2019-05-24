#DATA PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


## SKLEARN 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE 

#STATS
import statsmodels.tsa.api as smt # time-series analysis package
import statsmodels.api

kaggle_prices = './kaggle_data/prices.csv'
bloomberg_data = './bloomberg_data.csv'
year_2010 = './year_2010.csv'
new_data = 'processed_data.csv'

''' DATA PREPROCESSING '''
#Load Data and Convert to Matrix form, Drop NA entries

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

'''df = pd.read_csv(new_data,index_col=0)
df = df.iloc[:,0:500]
old_df = df'''

''' Data normalization '''
price_df = df 
tomorrow_price = df.iloc[:,1:]
today_price = df.iloc[:,0:(df.shape[1]-1)]
df = pd.DataFrame((tomorrow_price.values-today_price.values)/today_price.values,columns=today_price.columns, index = today_price.index)
df_mean = df.mean(axis=1)
df_std = df.std(axis=1) 
df = df.sub(df_mean,axis=0)
df = df.div(df_std,axis=0)
print(type(df))
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
    #print(possible_choices)
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

no_factors = onatski_factors()

loading_mat = eig_vecs[:,:no_factors]
factors = loading_mat.T@df.values


''' CLUSTERING BASED ON LOADING'''


# CHOICE 1: K-MEANS CLUSTERING 
no_groups = 20
model = KMeans(no_groups)
model.fit(loading_mat)
preds = model.predict(loading_mat)
labels = model.labels_
#print(preds)
#print(len(preds))

'''
Takes an array of stock group predictions
Returns an array of group arrays, which each contains the indices
of stocks in that group
'''
def get_groups(preds):

    # construct empty groups
    groups = []
    for i in range(no_groups):
        groups.append([])

    # assign stocks to groups
    for i,stock in enumerate(preds):
        groups[stock].append(i)
    
    s = 0
    for group in groups:
        s += len(group)
        #print(len(group))

    #print('sum')
    #print(s)
    # print(groups)
    return groups

groups = get_groups(preds)
#print(get_groups(preds))

# CHOICE 2: DBSCAN
dbscan_cluster = DBSCAN(eps=2.5*10e-3).fit(loading_mat)
new_labels = dbscan_cluster.labels_
#print(np.max(new_labels))
#print(new_labels)



''' TSNE VISUALIZATION ''' 
tsne_result = TSNE().fit_transform(loading_mat)
plt.figure(1, facecolor='white')
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=50, alpha=0.5,c=labels)
plt.savefig('tsne_kmeans')


plt.figure(1, facecolor='white')
plt.clf()
plt.scatter(tsne_result[(new_labels!=-1), 0], tsne_result[(new_labels!=-1), 1], s=50, alpha=0.5,c=new_labels[new_labels!=-1])
plt.savefig('tsne_dbscan')

def OU_residual_score(ts_1, ts_2):
    ts_1 = np.asarray(ts_1)
    ts_2 = np.asarray(ts_2)
    LR_model = LinearRegression().fit(ts_2.reshape(-1,1),ts_1)
    res_ts= ts_1 - (LR_model.coef_)*ts_2
    model = statsmodels.tsa.api.ARMA(res_ts,order=(1,0)).fit(disp=False)
    #print(model.summary())
    p_values = model.pvalues[1]
    coefs = model.params
    #p_values = statsmodels.stats.diagnostic.acorr_ljungbox(model.resid)[1]
    #res_score = np.min(p_values)
    return p_values, coefs 

score_matrices = []
good_pairs = []
for i in range(no_groups):
    this_group = groups[i]
    group_len = len(this_group)
    score_matrix = np.zeros((group_len,group_len))
    for j in range(group_len):
        for k in range(j+1,group_len):
            #print(j,k)
            OU_fit_result = OU_residual_score(df.iloc[this_group[j],:],df.iloc[this_group[k],:])
            score_matrix[j,k] = OU_fit_result[0]
            if (score_matrix[j,k]<0.05 and OU_fit_result[1][1]<1 and OU_fit_result[1][1]>0):
                good_pairs.append((this_group[j],this_group[k]))
    score_matrices.append(score_matrix)
 
print(len(good_pairs))



test_df = df
test_price_df = price_df

'''STAGE 2'''
slen = -1.25
slex = -0.5
ssen = 1.25
ssex = 0.75

def trading(stock_1,stock_2,train=60,trade=30,delta=1/252, interest= 0.02):
    ts_1 = np.asarray(test_df.iloc[stock_1,:])
    ts_2 = np.asarray(test_df.iloc[stock_2,:])
    price_1 = np.asarray(test_price_df.iloc[stock_1,:])
    price_2 = np.asarray(test_price_df.iloc[stock_2,:])
    t=train
    initial_wealth = 1.
    duration = len(ts_1)-train+1
    q_stock_1 = np.zeros(duration)
    q_stock_2 = np.zeros(duration)
    wealth = np.full(duration,initial_wealth)
    bank = wealth
    print(type(bank[0]))
    cash = initial_wealth
    while(t+trade<len(ts_1)):
        train_ts_1 = ts_1[t-train:t]
        train_ts_2 = ts_2[t-train:t]
        LR_model = LinearRegression().fit(train_ts_2.reshape(-1,1),train_ts_1)
        beta = LR_model.coef_
        train_res= train_ts_1 - (beta)*train_ts_2
        model = statsmodels.tsa.api.ARMA(train_res,order=(1,0)).fit(disp=False)
        a, b = model.params
        xi = model.resid 
        kappa= -np.log(b)/delta
        mean = a/(1-b)
        sigmasq = np.var(xi)*2*kappa/(1-b**2) 
        sigmaeq = np.sqrt(sigmasq/(2*kappa))
        for i in range(trade):
            if t+i>train:
                q_stock_1[t+i-train] = q_stock_1[t+i-1-train]
                q_stock_2[t+i-train] = q_stock_2[t+i-1-train]
                cash = bank[t+i-1-train]*((1+0.02/252)**(1/252))
            signal = ((ts_1[t+i]-(beta)*ts_2[t+i])-mean)/sigmaeq
            if signal > ssen:
                if q_stock_1[t+i-train] ==0:
                    q_stock_1[t+i-train] -= 1
                    q_stock_2[t+i-train] += beta
                    cash = cash + price_1[t+i] - beta*price_2[t+i]
            elif (signal <ssex and signal > slex):
                cash = cash + q_stock_1[t+i-train]*price_1[t+i] + q_stock_2[t+i-train]*price_2[t+i]
                q_stock_1[t+i-train] = 0
                q_stock_2[t+i-train] = 0 
            elif (signal < slen): 
                if q_stock_1[t+i-train] == 0:
                    q_stock_1[t+i-train] += 1
                    q_stock_2[t+i-train] -= beta
                    cash = cash - price_1[t+i] + beta*price_2[t+i]
            bank[t+i-train] = cash 
            wealth[t+i-train] = cash+ q_stock_1[t+i-train]*price_1[t+i]+q_stock_2[t+i-train]*price_2[t+i]
        t=t+10
    return wealth 


wealth=trading(good_pairs[1][0],good_pairs[1][1])
print(wealth)
plt.figure()
plt.plot(wealth)
plt.show()


            


