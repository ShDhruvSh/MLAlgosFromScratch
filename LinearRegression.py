import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionOneVar():

    def  __init__(self, x, y, alpha=0.01):
        self.alpha = alpha
        self.theta = np.zeros((len(x)+1, 1))
        self.x = np.matrix(x)
        self.average = np.average(self.x, axis=1)
        self.standardDev = np.std(self.x, axis=1)
        self.x = (self.x-self.average)/self.standardDev
        self.x = np.append(np.ones((1, self.x.shape[1])), self.x, axis=0)
        self.y = np.array(y)
        self.JVals = []

    def loss_function(self, theta, x, y):
        return (1/(2*x.shape[1]))*np.sum(np.square((theta.T@x) - y), axis=1)[0, 0]

    def batch_gradient_decent_fit(self, theta, x, y, epochs):
        for i in range(epochs):
            self.JVals.append(self.loss_function(theta, x, y))
            if i%100==0:
                print(f'Weights at Epoch {i}: \n{theta}')
            theta -= self.alpha*(1/x.shape[1])*np.sum(x@((theta.T@x) - y).T, axis=1)
        self.theta = theta
        return theta

    def normal_equation_fit(self, theta, x, y):
        self.theta = np.linalg.inv(x@x.T)@(x@y[:, None])
        return self.theta

    def gd_fit(self, epochs=1500):
        return self.batch_gradient_decent_fit(self.theta, self.x, self.y, epochs)

    def normal_fit(self):
        return self.normal_equation_fit(self.theta, self.x, self.y)

    def predict(self, input):
        return ((np.array(input)-self.average)/self.standardDev)*weights

df = pd.read_csv("./data/ex1data2.txt", header=None, names=['Size', 'NumBedrooms', 'Price'])
x = df.iloc[:, :-1].to_numpy().T
y = df.iloc[:, -1].to_numpy()

#Running the Model
model = LinearRegressionOneVar(x, y, alpha=0.1)
print(f'Loss Function Initial Value: {model.loss_function(model.theta, model.x, model.y)}')
gd_weights = model.gd_fit(epochs=50).flatten()
print(f'Final Gradient Decent Weights: \n{gd_weights}')
print(f'Loss Function Final Value for Gradient Decent: {model.loss_function(model.theta, model.x, model.y)}')
normal_weights = model.normal_fit().flatten()
print(f'Final Normal Equation Weights: \n{normal_weights}')
print(f'Loss Function Final Value for Normal Equation: {model.loss_function(model.theta, model.x, model.y)}')

#Plotting Outputs of Loss Function
plt.figure(figsize=(10, 10))
plt.plot(np.arange(50), np.array(model.JVals), label='Loss')
plt.xlabel('Number of Iterations')
plt.ylabel('Loss J')
plt.show()
