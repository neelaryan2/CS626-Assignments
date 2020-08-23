import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
from SVM import *

data = 'testing/2d_dataset_{}.csv'
df = pd.read_csv(data.format(1))

Y = df.iloc[:, 0:1].to_numpy()
X = df.iloc[:, 1:].to_numpy()
Y[Y == 0] = -1
X = min_max_scale(X)
X_train, Y_train, X_test, Y_test = split_data(*shuffle(X, Y))
mn = np.mean(X_train, axis=0)
X_train = add_ones(X_train - mn)
X_test = add_ones(X_test - mn)

model = SVM_Binary()
# model = SVM_Multi()
losses = model.fit(X_train, Y_train, lr=0.01, reg=0.0001, epochs=300, batch_size=1000, verbose=True)
Y_pred = model.predict(X_test)
mask = (Y_pred == Y_test)
# plt.plot(losses)
# plt.show()
print(np.mean(mask) * 100)


# visualize(X_train, Y_train)
# plt.show()



