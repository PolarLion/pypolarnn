import numpy 
import theano
import operator


class SimpleRNN:
  def __init__(self, input_dim=10, hidden_dim=10, out_dim=10, bptt_truncate=4):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    U = numpy.random.uniform(-numpy.sqrt(1./input_dim), numpy.sqrt(1./word_dim), (hidden_dim, input_dim))
    V = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (out_dim, hidden_dim))
    W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
    self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
    self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    self.theano = {}
    self.__theano.build__()

  def __theano_build__(self):
    U, V, W = self.U, self.V, self.W
    x = T.ivector('x')
    y = T.ivector('y')
    def forward_prop_step(x_t, s_t_prev, U, V, W):
      s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
      o_t = T.nnet.softmax(V.dot(s_t))
      return [o_t[0], s_t]
    [o,s], updates = theano.scan(
        forward_prop_step,
        sequences=x,
        outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
        non_sequences=[U, V, W],
        truncate_gradient=self.bptt_truncate,
        strict=True)

