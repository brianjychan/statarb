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

no_groups = 20
slen = -1.25
slex = -0.5
ssen = 1.25
ssex = 0.75

def get_data():
    year_2010 = './year_2010.csv'
    year_2011 = './year_2011.csv'

    df_2010 = data_process(year_2010)
    df_2011 = data_process(year_2011)
    return df_2010, df_2011

def data_process(year_2010):
    ''' DATA PREPROCESSING '''
    # Load Data and Convert to Matrix form, Drop NA entries
    df_2010 = pd.read_csv(year_2010)
    df_2010 = df_2010.drop('index', axis=1)
    df_2010['date'] = pd.to_datetime(df_2010['date'], format='%Y-%m-%d')
    df_2010 = df_2010.pivot(index='symbol', columns='date', values='price')
    df_2010 = df_2010.dropna()

    ''' Data normalization '''
    tomorrow_price = df_2010.iloc[:, 1:]
    today_price = df_2010.iloc[:, 0:(df_2010.shape[1] - 1)]
    df_2010 = pd.DataFrame((tomorrow_price.values - today_price.values) / today_price.values,
                           columns=today_price.columns, index=today_price.index)
    df_2010_mean = df_2010.mean(axis=1)
    df_2010_std = df_2010.std(axis=1)
    df_2010 = df_2010.sub(df_2010_mean, axis=0)
    df_2010 = df_2010.div(df_2010_std, axis=0)
    return df_2010

def pca(df):

    ''' PRINCIPAL COMPONENT ANALYSIS '''
    pca = PCA()
    pca.fit(df.T)
    eig_vecs = pca.components_.T
    eig_vals = pca.explained_variance_

    return (eig_vecs, eig_vals, pca)


# CRITERIA 1: THRESHOLD 
def threshold_factors(pca, threshold = 0.55):
    cumulated_variance = np.cumsum(pca.explained_variance_ratio_)
    possible_choices = np.argwhere(cumulated_variance>threshold)
    no_factors = possible_choices[0][0]+1
    return no_factors

# CRITERIA 2: BAI AND NG INFORMATION CRITERION
def bai_ng_factors(eig_vecs):
    N, T = eig_vecs.shape
    mse = np.zeros(T)
    counting_array = np.array((range(T))) #((N+T)/(N*T))*np.log((N*T)/(N+T))
    penalty = counting_array*(N+T - counting_array)*np.log(N*T)/(N*T)
    for i in range(T):
        loading_mat = eig_vecs[:,:i]
        factors = loading_mat.T@df.values
        mse[i] = mean_squared_error(df.values,loading_mat@factors)
    info_crit = mse + penalty
    no_factors = np.argmin(info_crit)+1
    return no_factors

# CRITERIA 3: ONATSKI
def onatski_factors(eig_vals, r_max = 15):
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
    



''' CLUSTERING BASED ON LOADING'''


def kmeans(loading_mat):
    # CHOICE 1: K-MEANS CLUSTERING 
    model = KMeans(no_groups)
    model.fit(loading_mat)
    preds = model.predict(loading_mat)
    labels = model.labels_
    return (preds, labels)

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


# CHOICE 2: DBSCAN
def dbscan(loading_mat):
    dbscan_cluster = DBSCAN(eps=2.5*10e-3).fit(loading_mat)
    new_labels = dbscan_cluster.labels_
    
    return new_labels


''' TSNE VISUALIZATION ''' 
def vis(loading_mat, labels):
    tsne_result = TSNE().fit_transform(loading_mat)
    plt.figure(1, facecolor='white')
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=50, alpha=0.5,c=labels)
    plt.title(label='Visualization of stock clusters using 2 factors')
    plt.savefig('2_factors')


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
    p_values = model.pvalues[1]
    coefs = model.params
    return p_values, coefs 

def get_score_matrix(df, groups):
    score_matrices = []
    good_pairs = []
    for i in range(no_groups):
        this_group = groups[i]
        group_len = len(this_group)
        score_matrix = np.zeros((group_len,group_len))
        for j in range(group_len):
            for k in range(j+1,group_len):
                OU_fit_result = OU_residual_score(df.iloc[this_group[j],:],df.iloc[this_group[k],:])
                score_matrix[j,k] = OU_fit_result[0]
                if (score_matrix[j,k]<0.05 and OU_fit_result[1][1]<1 and OU_fit_result[1][1]>0):
                    good_pairs.append((this_group[j],this_group[k]))
        score_matrices.append(score_matrix)

    return good_pairs, score_matrices
 

#####

'''STAGE 2'''


def trading(stock_1,stock_2, test_df, test_price_df, train=60,trade=10,delta=1/252, interest= 0.02):
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
    cash = initial_wealth
    kappa = None
    while(t+trade<len(ts_1)):
        train_ts_1 = ts_1[t-train:t]
        train_ts_2 = ts_2[t-train:t]
        LR_model = LinearRegression().fit(train_ts_2.reshape(-1,1),train_ts_1)
        beta = LR_model.coef_
        train_res= train_ts_1 - (beta)*train_ts_2
        model = statsmodels.tsa.api.ARMA(train_res,order=(1,0)).fit(disp=False)
        a, b = model.params
        xi = model.resid 
        previous_kappa = kappa 
        if b>0:
            kappa= -np.log(b)/delta
        else: 
            if previous_kappa is None:
                kappa = 10e5
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
        t=t+trade
    return wealth, q_stock_1, train_res

def sharpe_ratio(wealth,no_months=9):
    duration = len(wealth)
    monthreturn = np.zeros(no_months)
    for month in range(1,no_months):
        monthreturn[month-1] = (wealth[duration-20*no_months+20*month-1]/wealth[duration-20*no_months+20*(month-1)-1])-1
    meanmonth = np.mean(monthreturn)
    sdmonth = np.std(monthreturn)
    sharpe = (meanmonth*12-0.02)/(np.sqrt(12)*sdmonth)
    return sharpe 


#####

def plot_trading_results(wealth, train_res,q_stock_1):
    plt.figure()
    plt.plot(train_res)
    plt.figure()
    plt.plot(wealth[:-2])
    plt.figure()
    plt.plot(q_stock_1)
    print(sharpe_ratio(wealth))
######

def get_good_sharpe_ratios(good_pairs, test_df, test_price_df):
    sharpe_ratios = np.zeros(len(good_pairs))
    for i in range(len(good_pairs)):
        wealth, q_stock_1, train_res=trading(good_pairs[i][0],good_pairs[i][1], test_df, test_price_df)
        sharpe_ratios[i] = sharpe_ratio(wealth)
    return sharpe_ratios

if __name__ == "__main__":
    df, price_df = get_data()

    eig_vecs, eig_vals, pca = pca(df)

    no_factors = bai_ng_factors(eig_vecs) 
    loading_mat = eig_vecs[:,:no_factors]
    factors = loading_mat.T@df.values

    preds, labels = kmeans(loading_mat)
    groups = get_groups(preds)
    new_labels = dbscan(loading_mat)
    vis(loading_mat, labels)
    good_pairs, score_matrices = get_score_matrix(df, groups)

    # Stage 2
    test_df = df
    test_price_df = price_df
    wealth, q_stock_1, train_res=trading(test_df, test_price_df, good_pairs[6][0],good_pairs[6][1])
    sharp_ratios = get_good_sharpe_ratios(good_pairs, test_df, test_price_df)
    plot_trading_results(wealth, train_res, q_stock_1)


    


    
