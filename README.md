# ChebyGCN

```python
pip install ChebyGCN
```

Notice, for training and testing data, permutations of the data must be done in a certain way to align with 
pooling of the graph lapacian. Further, every level of graph corsening is a pool of size two, thus if you want to 
pool by 2 and then 4, you need log_2(2 * 4)= 3 levels. You will also need to index your Lapancians as seen below.

```python 
from ChebyGCN import layers, coarsening
A = scipy.sparse.csr.csr_matrix(A) #load adjanecy matrix 
graphs, perm = coarsening.coarsen(A, levels=3, self_connections=True) #produce graph coarsenings 
X_train = coarsening.perm_data(X_train, perm)
X_test = coarsening.perm_data(X_test, perm)
L = [coarsening.laplacian(A, normalized=True) for A in graphs]

x_input = Input(shape=(X_train.shape[1],))
x = Reshape((X_train.shape[1],1))(x_input)
x = layers.GraphConvolution( 8, 2, 20, L[0])(x)
x = layers.GraphConvolution( 8, 4, 10, L[2])(x)
x = Flatten()(x)
x = Dense(66, activation='softmax')(x)
```


This code is 96% based on https://github.com/mdeff/cnn_graph Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, Neural Information Processing Systems (NIPS), 2016.