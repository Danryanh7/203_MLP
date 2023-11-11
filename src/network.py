# To understand the architecture of MLPs...
import numpy as np
import random
from types import List
from utils import sigmoid, dSigmoid

class Network(object):
    """
    Network class for the MLP
    Attributes:
        nLayers: Number of layers
        sizes: Neurons per layer
        biases: Biases per layer (randomly initialized)
        weights: Weights per layer (randomly initialized)

        It should be noted that the first layer is the input layer
        so it does not have biases nor weights
    """
    def __init__(self, sizes: List):
        self.nLayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedForward(self, x):
        """
        Returns the output of the network if x is the input
        """
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(np.dot(w, x) + b)
        output = x
        return output
    
    def dCost(self, oA, y):
        """
        Returns a vector of partial(Cx)/partial(a) for the output activations
        """
        return (oA-y)

    def stochastic_gradient_descent(self, xTrain: tuple, epochs: int, batchSize: int, nu: int, xTest: tuple=None):
        """
        Implementing stochastic gradient descent
        xTrain: Tuple of the training input and the desired output
        epochs: Number of epochs
        batchSize: Size of the batch
        nu: Learning rate
        xTest: Tuple of test input and it's correspodning output for verification
        """
        if xTest:
            nTest = len(xTest)
        n = len(xTrain)
        for i in range(epochs):
            # Randomly shuffle the training data
            random.shuffle(xTrain)
            # Partition into batches
            batches = [xTrain[k:k+batchSize] for k in range(0, n, batchSize)]
            # For each batch apply gradient descent
            for batch in batches:
                # Apply gradient descent
                self.updateBatch(batch, nu)
            if xTest:
                print(f"Epoch {i}: {self.evaluate(xTest)} / {nTest}")
            else:
                print(f"Epoch {i} complete")

    def updateBatch(self, batch: List[tuple], nu: int):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single batch
        batch: a list of tuples (x, y) representing the training inputs and the desired outputs
        nu: the learning rate
        """
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            # Returns the gradient for the cost associated with each training example
            deltaNablaB, deltaNablaW = self.backpropagation(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]
        self.weights = [w - (nu / len(batch)) * nw for w, nw in zip(self.weights, nablaW)]
        self.biases = [b - (nu / len(batch)) * nb for b, nb in zip(self.biases, nablaB)]

    def backpropagation(self, x, y):
        """
        Returns nablaB and nablaW representing components for the gradient for the cost function Cx.
        """
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        # Forward Pass
        a = x
        aList = [x] # Stores all the activations, layer by layer

        zs = [] # Stores all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            aList.append(a)
        
        # Backward Pass
        delta = self.dCost(aList[-1], y) * dSigmoid(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, aList[-2].transpose())

        for l in range(2, self.nLayers):
            z = zs[-l]
            ds = dSigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * ds
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, aList[-l-1].transpose())
        return (nablaB, nablaW)