import random
import numpy as np


class Network(object):
    def __init__(self, sizes, identifier):
        #"""The list ``sizes`` contains the number of neurons in the
        # respective layers of the network.  For example, if the list
        # was [2, 3, 1] then it would be a three-layer network, with the
        # first layer containing 2 neurons, the second layer 3 neurons,
        # and the third layer 1 neuron.  The biases and weights for the
        # network are initialized randomly, using a Gaussian
        # distribution with mean 0, and variance 1.  Note that the first
        # layer is assumed to be an input layer, and by convention we
        # won't set any biases for those neurons, since biases are only
        # ever used in computing the outputs from later layers."""
        self.id = identifier
        self.num_layers = len(sizes)
        self.sizes = sizes
        print 'making stuff'
        self.biases = [np.full((y, 1),0.0) for y in sizes[1:]]
        self.weights = [(np.random.randn(y, x) * (2.0/x)**.5) for x, y in zip(sizes[:-1], sizes[1:])]


    def equals(self, net2):
        if self.id != net2.id:
            print 'diff id'
            return False
        for x,y in zip(self.sizes, net2.sizes):
            if x != y:
                print 'diff size'
                return False
        for i in range(0,len(self.biases)):
            if not np.array_equal(self.biases[i], net2.biases[i]):
                print 'diff bias'
                return False
        for i in range(0,len(self.weights)):
            if not np.array_equal(self.biases[i], net2.biases[i]):
                print 'diff weight'
                return False

        return True



    def copy(self):
        newNet = Network(self.sizes , self.id)
        newNet.num_layers = self.num_layers
        newNet.sizes = [x for x in self.sizes]
        for i in range(0,len(self.biases)):
            newNet.biases[i] = self.biases[i]
        for i in range(0,len(self.weights)):
            newNet.weights[i] = self.weights[i]
        return newNet


    def feedforward(self, a):
        
       # """Return the output of the network if ``a`` is input."""
       

        for b, w in zip(self.biases, self.weights):
            
            a= act(np.dot(w,a)+ b, self.id)
             
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        #"""Train the neural network using mini-batch stochastic
        # gradient descent.  The ``training_data`` is a list of tuples
        #``(x, y)`` representing the training inputs and the desired
        # outputs.  The other non-optional parameters are
        # self-explanatory.  If ``test_data`` is provided then the
        # network will be evaluated against the test data after each
        # epoch, and partial progress printed out.  This is useful for
        # tracking progress, but slows things down substantially."""
        init_weights = []
        location = 0
        for i in range(0,len(self.weights)):
            init_weights.append(self.weights[i])
        print init_weights[location]
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        x = self.weights[-1]
        y = self.weights[0]
        error = 0
        g = 0
        for j in xrange(epochs):
            #print 'eeey'
            if (j % 3 == 0):
                self.deadCheck(training_data)
            err = self.evaluate(training_data)
            print err
            # if error != err:
            #     error = err
            #     print err
            #     eta *= 1.3
            # else:
            #     error = err
            #     print err
            #     eta /= 1.5
            #     print 'eta too high'
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                g = self.update_mini_batch(mini_batch, eta)[-1]#
            #print (self.weights[-1] - x) #/ x
            #print (self.weights[0] - y) #/ y
            #if test_data:
             #   print "Epoch {0}: {1} / {2}".format(
              #      j, self.evaluate(test_data), n_test)
            
            print "Epoch {0} complete".format(j)
        print g
        print 'a'

        #print (init_weights[location] - self.weights[location]) 
    def update_mini_batch(self, mini_batch, eta):
       # """Update the network's weights and biases by applying
        # gradient descent using backpropagation to a single mini batch.
        # The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        # is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #print 'im here!!'
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
       
        
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        return nabla_w
        #print self.weights[0][0]
        #print ' '
        #print ' '
        #print ' '
        #print ' '

        

    def backprop(self, x, y):
        
     #   """Return a tuple ``(nabla_b, nabla_w)`` representing the
     #   gradient for the cost function C_x.  ``nabla_b`` and
     #   ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
     #   to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = act(z,self.id)
            activations.append(activation)
        #print activations[-1]
        #print y

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            act_prime(zs[-1], self.id)
        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = act_prime(z,self.id)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            #print delta
            #print ' '
            #print activations[-l-1].transpose()
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        for l in xrange(1,self.num_layers):
           pass 
          # print nabla_w[-l]

        #print ' '

        return (nabla_b, nabla_w)

    def deadCheck(self,test_data):
        activations = [np.full((x,1), 0.0) for x in self.sizes[1:]]
        for a,q in test_data:
            ite = 0
            for b, w in zip(self.biases, self.weights):

                a= act(np.dot(w,a)+ b,self.id)
                activations[ite] += a
                ite+=1
        dead = 0
        maxi = 0
        for a in activations:
            for b in a:
                for c in b:
                    maxi+=1
                    if c == 0:
                        dead +=1
        f = True
        # f = False if x < 0 for x in activations[-1]
        for val in activations[-1]:
            if val != 0.0:
                f = False
        if f:
            print 'AAAAAAAAAAAAAAAA'
        if f:
            print 'CHANGING WEIGHTS...'
            self.biases = [np.full((y, 1),-.01) for y in self.sizes[1:]]
            self.weights = [(np.random.randn(y, x) * (2.0/y)**.5) for x, y in zip(self.sizes[:-1], self.sizes[1:])]



        print dead
        print maxi





             
        

    def evaluate(self, test_data):
     #   """Return the number of test inputs for which the neural
     #   network outputs the correct result. Note that the neural
     #   network's output is assumed to be the index of whichever
     #   neuron in the final layer has the highest activation."""
        
        return sum((y - self.feedforward(x))**2 for x,y in test_data)

    def cost_derivative(self, output_activations, y):
      #  """Return the vector of partial derivatives \partial C_x /
      #  \partial a for the output activations."""
        return (output_activations - y)

# Miscellaneous functions

def act(z, i):
    if i == 0:
        return sigmoid(z)
    elif i== 1:
        return relu(z)
    elif i == 2:
        return lRelu(z)
    else:
        print 'error with net id'


def act_prime(z, i):
    if i == 0:
        return sigmoid_prime(z)
    elif i == 1:
        return relu_prime(z)
    elif i == 2:
        return lRelu_prime(z)
    else:
        print 'error with net id'



def sigmoid(z):
 #   """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return z * (z>0)

def relu_prime(z):
    return (z>0) * 1 


def sigmoid_prime(z):
 #   """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def lRelu(z):
    return z * (z>0) + .01 * z * (z <= 0)

def lRelu_prime(z):
    return (z>0) * 1 + (z<=0) * (-.01)





