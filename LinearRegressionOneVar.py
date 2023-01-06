import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


class LinearRegressionOneVar():

    def  __init__(self, x, y, alpha=0.01):
        self.alpha = alpha
        self.theta = np.zeros((2, 1))
        self.x = np.append(np.ones((1, len(x))), np.matrix(x), axis=0)
        self.y = np.array(y)
        self.oldThetas = []

    
    def loss_function(self, theta, x, y):
        return (1/(2*x.shape[1]))*np.sum(np.square((theta.T@x) - y), axis=1)

    def batch_gradient_decent_fit(self, theta, x, y, epochs):
        for i in range(epochs):
            if i%50==0:
                self.oldThetas.append(theta.flatten())
            if i%100==0:
                print(f'Weights at Epoch {i}: \n{theta}')
            theta -= self.alpha*(1/x.shape[1])*np.sum(x@((theta.T@x) - y).T, axis=1)
        self.theta = theta
        return theta

    def fit(self, epochs=1500):
        return self.batch_gradient_decent_fit(self.theta, self.x, self.y, epochs)


df = pd.read_csv("./data/ex1data1.txt", header=None, names=['Population', 'Profit'])
x = df['Population'].to_numpy()
y = df['Profit'].to_numpy()

fig = plt.figure(figsize=(2, 2))
data = fig.add_subplot(221)

data.scatter(x, y) 
data.set_xlabel('Population of City in 10,000s')
data.set_ylabel('Profit in $10,000s')
data.title.set_text('Data With Regression Line')

#Running the Model
model = LinearRegressionOneVar(x, y)
print(f'Loss Function Initial Value: {model.loss_function(model.theta, model.x, model.y)}')
weights = model.fit().flatten()
print(f'Final Weights: \n{weights}')
print(f'Input [1, 3.5]: {np.array([1, 3.5]) * weights}')
print(f'Input [1, 7]: {np.array([1, 7]) * weights}')

data.plot(x, x*weights[1] + weights[0], label='Line of Best Fit')
leg = data.legend(loc='upper left')

#Surface and Contour Plots
surfacePlot = fig.add_subplot(222, projection='3d')
contourPlot = fig.add_subplot(2, 2, (3, 4))

loss_vals = np.zeros((len(model.oldThetas), len(model.oldThetas)))
npOldThetas = np.array(model.oldThetas)
plottedThetas = []
for i, theta0 in enumerate(npOldThetas[:, 0]):
    for j, theta1 in enumerate(npOldThetas[:, 1]):
        currTheta = np.array([theta0, theta1])
        plottedThetas.append(currTheta)
        loss_vals[i, j] = model.loss_function(currTheta.reshape(2, 1), model.x, model.y)
plottedThetas = np.array(plottedThetas)

surfacePlot.plot_trisurf(plottedThetas[:, 0], plottedThetas[:, 1], loss_vals.flatten(), linewidth=0, antialiased=False)
surfacePlot.set_xlabel('Theta0s')
surfacePlot.set_ylabel('Theta1s')
surfacePlot.set_zlabel('Losses')
surfacePlot.title.set_text('Surface Plot')

contourPlot.tricontourf(plottedThetas[:, 0], plottedThetas[:, 1], loss_vals.flatten(), linewidth=0, antialiased=False)
contourPlot.set_xlabel('Theta0s')
contourPlot.set_ylabel('Theta1s')
contourPlot.title.set_text('Contour Plot')

plt.show()
