import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Add, ReLU, Input, Flatten
from tensorflow.keras.models import Model
from typing import NamedTuple, List

class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: tf.Tensor # Dict[Action, float]
    hidden_state: List[float] # not sure about this one lol
        
        
class Network(object):
    def __init__(self, h_in=5, w_in=5, c_in=32, nf=128, n_acts=4):
        # Initialise a uniform network - should I init these networks explicitly?
        self.f = PredNet((5,5,nf), nf, n_acts)
        self.g = DynaNet((5,5,nf+1), nf)
        self.h = ReprNet((80,80,32), nf)
        self.steps = 0

#     @tf.function
    def initial_inference(self, obs) -> NetworkOutput:
        # representation + prediction function
        # input: 32x80x80 observation
        state = self.h(obs)
        policy_logits, value = self.f(state)
        # state, policy_logits, value = self.initial_inference_compiled(obs)
        # policy = {Action(i):p for i,p in enumerate(policy_logits[0])}
        policy = policy_logits[0]
        return NetworkOutput(value[0], 0.0, policy, state) # keep state 4D
    
#     @tf.function
    def recurrent_inference(self, state, action) -> NetworkOutput:
        # dynamics + prediction function
        # Input: hidden state nfx5x5
        # Concat/pad action to channel dim of states
        state_action = tf.pad(state, paddings=[[0, 0], [0, 0], [0, 0], [0, 1]], constant_values=action)
        next_state, reward =  self.g(state_action)
        policy_logits, value = self.f(next_state)
        # next_state, reward, policy_logits, value = self.recurrent_inference_compiled(state, action)
        # policy = {Action(i):p for i,p in enumerate(policy_logits[0])}
        policy = policy_logits[0]
        return NetworkOutput(value[0], reward[0], policy, next_state)

    def get_weights(self):
        # Returns the weights of this network.
        self.steps += 1 # probably not ideal
        return [self.f.parameters(), self.g.parameters(), self.h.parameters()]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
        
    
### Tensorflow model definitions ###

def ResBlock(x_in, nf=128):
    x = Conv2D(nf, 3, padding='same', use_bias=False)(x_in)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(nf, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = ReLU()(x)
    return x

def ConvBlock(x_in, nf, s=1, bn=True):
    x = Conv2D(nf, 3, padding='same', strides=s, use_bias=not bn)(x_in)
    if bn: x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def ReprNet(input_shape=(80,80,32), nf=128):
    o = Input(shape=input_shape)
    x = ConvBlock(o, nf, 2)
    x = ConvBlock(x, nf, 2)
    x = ConvBlock(x, nf, 2)
    s = ConvBlock(x, nf, 2)
    return Model(o, s)

def DynaNet(input_shape=(5,5,129), nf=128):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf)
    x = ConvBlock(x, nf)
    s_new = ConvBlock(x, nf)
    
    r = Flatten()(s_new)
    r = Dense(1)(r)
    return Model(s, [s_new, r])

def PredNet(input_shape=(5,5,128), nf=64, num_actions=4):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf//2) 
    x = Flatten()(x)

    a = Dense(num_actions)(x)
    v = Dense(1)(x)
    return Model(s, [a, v])
