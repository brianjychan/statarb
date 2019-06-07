import numpy as np 

FILEPATH = "./random_sharpe_10k.txt"
arr = np.loadtxt(FILEPATH)

print(arr[:5])