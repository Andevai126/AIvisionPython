import numpy as np
from data import *



# Prepare data
def flattenGrid(grid):
    return [bit for row in grid for bit in row]
def transpose(xs):
    return np.array([[x] for x in xs])
dataTrain = [(transpose(flattenGrid(grid)), transpose(outputDict[label])) for grid, label in trainingSet]
dataTest = [(transpose(flattenGrid(grid)), transpose(outputDict[label])) for grid, label in testSet]



class NN:
    def __init__(self, layers):
        self.nLayers = len(layers)
        # Weigts initialized random: mean 0, variance 1, standard deviation 0.01
        self.Ws = [np.random.randn(y, x) * 0.01 for x, y in zip(layers[:-1], layers[1:])]
        # Biases initialized to zero
        self.Bs = [np.zeros((y, 1)) for y in layers[1:]]
    
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoidPrime(self, x):
        sig = self.sigmoid(x)
        return sig * (1.0 - sig)
    
    def ReLU(self, x):
        return np.maximum(x, 0)
    
    def ReLUPrime(self, x):
        return x > 0
    
    def softmax(self, x):
        max = np.max(x)
        exps = np.exp(x - max)
        sum = np.sum(exps)
        return exps / sum
    
    def softmaxPrime(self, x):
        s = self.softmax(x)
        # Return Jacobian matrix
        return np.diagflat(s) - np.dot(s, s.T)
    
    def forwardProp(self, X):
        Zs = []
        As = [X]
        A = X

        # Hidden layers
        for W, B in zip(self.Ws, self.Bs):
            Z = np.dot(W, A) + B
            A = self.ReLU(Z) # Replace ReLU() with sigmoid() for sigmoid in hidden layers
            Zs.append(Z)
            As.append(A)
        
        # Output layer
        As[-1] = self.softmax(Zs[-1]) # Remove for sigmoid in last layer

        return (Zs, As)

    def backwardProp(self, Zs, As, Y):
        nablaWs = [np.zeros(W.shape) for W in self.Ws]
        nablaBs = [np.zeros(B.shape) for B in self.Bs]

        # Output layer
        # delta = 2 * (As[-1] - Y) * self.sigmoidPrime(Zs[-1]) # Activate for sigmoid in last layer
        # delta = np.dot(self.softmaxPrime(Zs[-1]), 2 * (As[-1] - Y)) # Activate for softmax in last layer
        delta = 2 * (As[-1] - Y) # Activate to skip softmax derivate of last layer
        nablaWs[-1] = np.dot(delta, As[-2].T)
        nablaBs[-1] = delta

        # Hidden layers
        for L in range(2, self.nLayers):
            delta = np.dot(self.Ws[-L+1].T, delta) * self.ReLUPrime(Zs[-L]) # Replace ReLUPrime() with sigmoidPrime() for sigmoid in hidden layers
            nablaWs[-L] = np.dot(delta, As[-L-1].T)
            nablaBs[-L] = delta
        
        return (nablaWs, nablaBs)

    def train(self, data, maxEpochs, learningRate, maxCost):
        for i in range(maxEpochs):
            totalCost = 0.0

            for X, Y in data:
                # Get summed weighted inputs and outputs of activation functions
                Zs, As = self.forwardProp(X)
                # Get gradients of weights and biases
                nablaWs, nablaBs = self.backwardProp(Zs, As, Y)
                # Update
                self.Ws = [Ws - learningRate*nw for Ws, nw in zip(self.Ws, nablaWs)]
                self.Bs = [Bs - learningRate*nb for Bs, nb in zip(self.Bs, nablaBs)]
                # Add MSE
                totalCost += np.mean((As[-1] - Y) ** 2)

            if totalCost < maxCost:
                print(f"Stopped early at epoch {i}")
                return i
        return maxEpochs

    def predict(self, X):
        Zs, As = self.forwardProp(X)
        V = np.zeros((len(As[-1]), 1))
        V[np.argmax(As[-1])] = 1.0
        # Return both raw and vectorized output
        return (As[-1], V)



nn = NN([9, 39, 2])
nn.train(dataTrain, 1000, 0.1, 0.01)
for X, Y in dataTest:
    A, V = nn.predict(X)
    # From top to bottom:
    # Expected, raw predicted, vectorized predicted, comparison
    print("Y: ", Y.flatten())
    print("A: ", A.flatten())
    print("V: ", V.flatten())
    print("C: ", np.array_equal(Y, V))
    print("")



# nFaults = 0
# nEpochs = 0
# for i in range(1000):
#     nn = NN([9, 39, 2])
#     nEpochs += nn.train(dataTrain, 1000, 0.1, 0.01)
#     for X, Y in dataTest:
#         A, V = nn.predict(X)
#         if (np.array_equal(Y, V) == False):
#             nFaults += 1
#     if i % 100 == 0 and i != 0:
#         print(f"Completed {i} epochs")
# print(f"Total of faults found: {nFaults}")
# print(f"Average epochs needed: {nEpochs/1000}")