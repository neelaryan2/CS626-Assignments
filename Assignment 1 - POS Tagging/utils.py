import numpy as np
import matplotlib.pyplot as plt

def shuffle(X, Y):
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    return X[idx], Y[idx]

def add_ones(X):
    ones = np.ones((X.shape[0], 1))
    return np.append(X, ones, axis=1)

def get_batches(X, Y, batch=32):
    N = X.shape[0]
    X_batch, Y_batch = [], []
    Q, R = N // batch, N % batch
    for i in range(Q):
        X_batch.append(X[i*batch:(i+1)*batch, :])
        Y_batch.append(Y[i*batch:(i+1)*batch])
    if R != 0 :
        X_batch.append(X[Q*batch:, :])
        Y_batch.append(Y[Q*batch:])
    return X_batch, Y_batch

def split_util(X, train_ratio=0.8):
    slic = int(X.shape[0] * min(max(train_ratio, 0.0), 1.0))
    X_train, X_test = X[:slic], X[slic:]
    return X_train, X_test

def split_data(X, Y, train_ratio=0.8):
    X_train, X_test = split_util(X, train_ratio)
    Y_train, Y_test = split_util(Y, train_ratio)
    return X_train, Y_train, X_test, Y_test

def one_hot_encode(X, labels):
    X.shape = (X.shape[0], 1)
    newX = np.zeros((X.shape[0], len(labels)))
    label_encoding = {}
    for i, l in enumerate(labels):
        label_encoding[l] = i
    for i in range(X.shape[0]):
        newX[i, label_encoding[X[i, 0]]] = 1
    return newX

def normalize(X):
    return (X - np.mean(X)) / np.std(X)

def untag(tagged_sent):
    return [word for word, tag in tagged_sent]

def preprocess(X, Y):
    N, D = X.shape
    X_new = [np.ones((N, ), dtype=np.float64)]
    for i in range(1, D):
        col = X[:, i]
        try:
            col = col.astype(np.float64)
            new_cols = normalize(col)
            X_new.append(new_cols)
        except ValueError:
            labels = sorted(list(set(col)))
            new_cols = one_hot_encode(col, labels)
            for j in range(new_cols.shape[1]):
                X_new.append(new_cols[:, j])
    X_new = np.array(X_new).T
    return X_new, Y

def visualize(X, Y, W=None):
    fig = plt.figure()
    for i in range(X.shape[0]):
        if Y[i][0] > 0 :
            plt.scatter(X[i][0], X[i][1], marker='+', color='red')
        else :
            plt.scatter(X[i][0], X[i][1], marker='_', color='blue')
    if W is None : return
    xmin, xmax = min(X[:, 0]), max(X[:, 0])
    linex = np.linspace(xmin, xmax, num=50)
    liney = [-(W[2] + W[0] * e) / W[1] for e in linex]
    plt.plot(linex, liney)

def min_max_scale(x):
    mn = np.min(x, axis=0)
    r = np.ptp(x, axis=0)
    mask = (r == 0)
    r[mask] = mn[mask]
    return (x - mn) / r