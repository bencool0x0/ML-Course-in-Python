import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def computeCost(X, y, theta):
    m = len(y)
    J = 0
    
    for i in range(m):
        J = J + (np.matmul(theta, X[i, :].transpose()) - y[i, :])**2
    
    J = J / (2 * m)

    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.empty([num_iters, 1])

    for iter in range(num_iters):
        temps = np.zeros([1, X.shape[1]])
        #row vector with two columns (check)

        #print("X.shape[1]", X.shape[1])
        for i in range(X.shape[1]):
            #for each parameter- two of them- (check)

            diff = 0   
            for j in range(m): 
                #for each training example

                xVector = (X[j, :])[np.newaxis, :] 
                #1x2 row vector

                #print("theta", (theta))
                #print("xvector", (xVector.transpose()))
                #print("z", np.matmul(theta, xVector.transpose()))
                #print("y", y[i,:])
                #print("x", X[j,i])

                diff = diff + ((np.matmul(theta, xVector.transpose()) - y[j, :]) * X[j,i])
                #print("example", j, "x", xVector, "y", y[j,:], "grad: ", ((np.matmul(theta, xVector.transpose()) - y[j, :]) * X[j,i]))
            
            temps[0,i] = theta[0,i] - (alpha / m) * diff

        theta = temps

        J_history[iter] = computeCost(X, y, theta)

    return theta, J_history

def readFile(fileName):
    data = np.genfromtxt(fileName, delimiter=',')
    data = data[1:]
    return data

def normalizeFeatures(matrix):
    m = matrix
    for i in range(1, np.shape(matrix)[1]):
        mu = np.mean(m[:,i])
        print("mean:", mu)
        m[:,i] = np.subtract(m[:, i], mu)
        sigma = np.std(m[:, i])
        print("std:", sigma)
        if sigma != 0:
            m[:, i] = np.divide(m[:, i], sigma)
    
    return m


def main():
   print("hi")
   data = readFile("test.csv")
   plt.plot(data[:, 0], data[:, 1])
   plt.show()
   data = normalizeFeatures(data)
   plt.plot(data[:, 0], data[:, 1])
   plt.show()
   a = data[:,0]
   a = a[:, np.newaxis]
   #making vector 2D

   X = np.concatenate((np.ones((np.shape(a)[0], 1)), a), axis = 1)
   #X with column of ones for intercept

   b = data[:,1]
   y = b[:, np.newaxis]
   #making vector 2D

   t = np.array([1, 0])
   t = t[np.newaxis, :]
   #making theta vector 2D

   a1 = 0.00005 #for test
   a2 = 0.052 #for salary data

   num_iters = 100
   trainedTheta, J_his = gradientDescent(X, y, t, a1, num_iters)
   #train the data

   plt.plot(np.arange(num_iters), J_his)
   plt.show()
   
   predictions = (np.zeros(np.shape(data)[0]))
   #trained on test set, so should match
   for i in range(np.shape(data)[0]):
    predictions[i] = np.matmul(trainedTheta, X[i, :].transpose())

   plt.plot(data[:,0], data[:,1], label= "examples")
   plt.plot(data[:,0], predictions, label = "predictions")
   plt.show()

   cost = computeCost(X, y, trainedTheta)
   print("cost", cost)
   print("WHY IS THE COST SO HIGH???")
   print("Regression line works though")


if __name__ == "__main__":
    main()