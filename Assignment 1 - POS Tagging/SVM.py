import numpy as np
from utils import *

class SVM_Binary:
    def __init__(self, model_path='./SVM_Binary.npz'):
        self.W = None
        if not model_path.endswith('.npz'):
            model_path += '.npz'
        self.path = model_path

    def loss(self, X, Y, reg):
        N = X.shape[0]
        d = np.maximum(0, 1 - Y * np.matmul(X, self.W))

        loss = np.sum(d) / N + reg * np.sum(np.square(self.W)) / 2

        M = ((d > 0) * Y * X)
        M = np.mean(M, axis=0, keepdims=True)
        dw = reg * self.W + M.T

        return loss, dw

    def fit(self, X, Y, lr=1e-3, reg=1e-5, epochs=100, batch_size=10, verbose=False):
        N, D = X.shape
        if self.W is None:
            self.W = 0.001 * np.random.standard_normal(size=(D, 1))
        
        loss_history = []
        for epoch in range(epochs):
            loss = 0.0
            X_batches, Y_batches = get_batches(*shuffle(X, Y), batch=batch_size)
            grad = np.zeros_like(self.W)
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                cur_loss, grad = self.loss(X_batch, Y_batch, reg)
                self.W -= lr * grad
                loss += cur_loss * X_batch.shape[0]
            loss_history.append(loss / N)
            if verbose and epoch % 100 == 0:
                print('Epoch {}/{} | Loss {}'.format(epoch, epochs, loss))
        return loss_history

    def predict(self, X):
        y_pred = np.sign(np.matmul(X, self.W))
        return np.expand_dims(y_pred, axis=1)

    def save(self):
        np.save(self.path, self.W)
    
    def load(self):
        self.W = np.load(self.path)


class SVM_Multi():
    def __init__(self, model_path='./SVM_Multi.npz'):
        self.W = None
        if not model_path.endswith('.npz'):
            model_path += '.npz'
        self.path = model_path

    def loss(self, X, Y, reg):
        dW = np.zeros_like(self.W) 
        N, _ = X.shape
        scores = np.matmul(X, self.W)

        correct = scores[np.arange(N), Y].reshape(N, 1)
        margin = np.maximum(0, scores - correct + 1)
        margin[np.arange(N), Y] = 0
        loss = np.sum(margin) / N + reg * np.sum(np.square(self.W))

        margin[margin > 0] = 1
        valid_margin_count = np.sum(margin, axis=1)
        margin[np.arange(N), Y] -= valid_margin_count
        dW = np.matmul(X.T, margin) / N + reg * 2 * self.W

        return loss, dW

    def fit(self, X, Y, lr=1e-3, reg=1e-5, epochs=100, batch_size=10, verbose=True):
        N, D = X.shape
        # NOTE : each label c should follow : 0 <= c < C where C is number of classes
        C = np.max(Y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.standard_normal(size=(D, C))
        loss_history = []
        for epoch in range(epochs):
            loss = 0.0
            X_batches, Y_batches = get_batches(*shuffle(X, Y), batch=batch_size)
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                cur_loss, grad = self.loss(X_batch, np.squeeze(Y_batch), reg)
                self.W -= lr * grad
                loss += cur_loss * X_batch.shape[0]
            loss_history.append(loss / N)
            if verbose and epoch % 100 == 0:
                print('Epoch {}/{} | Loss {}'.format(epoch, epochs, loss))
        return loss_history

    def predict(self, X):
        scores = np.matmul(X, self.W)
        y_pred = scores.argmax(axis=1)
        return np.expand_dims(y_pred, axis=1)

    def save(self):
        np.save(self.path, self.W)
    
    def load(self):
        self.W = np.load(self.path)