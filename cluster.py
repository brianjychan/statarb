from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


        
def knn(loading_mat):
    neigh = NearestNeighbors(20, 1.0)
    neigh.fit(loading_mat)
    distances, indices = neigh.kneighbors(loading_mat)
    print("indices: ")
    print(indices)
    print(len(indices))
    print(len(indices[0]))
    return indices


def kmeans(loading_mat):
    model = KMeans(20)
    model.fit(loading_mat)
    preds = model.predict(loading_mat)
    print(preds)
    print(len(preds))
    return preds

'''
Takes an array of stock group predictions

Returns an array of group arrays, which each contains the indices
of stocks in that group
'''
def get_groups(preds):

    # construct empty groups
    groups = []
    for i in range(20):
        groups.append([])

    # assign stocks to groups
    for i,stock in enumerate(preds):
        groups[stock].append(i)
    
    s = 0
    for group in groups:
        s += len(group)
        print(len(group))

    print('sum')
    print(s)
    # print(groups)
    return groups