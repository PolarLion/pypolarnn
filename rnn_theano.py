import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
import operator

class RNNTheano:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        b = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),size=hidden_dim)
        c = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),size=word_dim)
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        print 'v', V.shape
        # Theano: Created shared variables
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))      
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        b, c = self.b, self.c
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, b, c, U, V, W):
            s_t = T.tanh(b + U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(c + V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[b, c, U, V, W],
            truncate_gradient=4,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        # o_error1 = T.sum(T.nnet.categorical_crossentropy(o, y))
        o_error = -T.sum(T.log(o)[T.arange(y.shape[0]), y])
        
        # Gradients
        db = T.grad(o_error, b)
        dc = T.grad(o_error, c)
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
       
        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [db, dc, dU, dV, dW])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [dc[0]], 
                      updates=[
                            (self.b, self.b - learning_rate * db),
                            (self.c, self.c - learning_rate * dc),
                            (self.U, self.U - learning_rate * dU),
                              (self.V, self.V - learning_rate * dV),
                              (self.W, self.W - learning_rate * dW)])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)   


