from sklearn.linear_model import LinearRegression
import numpy as np

def trade(spread):
    today = spread[:-1]
    tomor = spread[1:]
    reg = LinearRegression().fit(today.reshape(-1,1), tomor)
    b, a = reg.coef_
    var_noise = np.var(reg._residues)
    return b,a,var_noise
