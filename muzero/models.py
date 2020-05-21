import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Add, ReLU, Input, Flatten
from tensorflow.keras.models import Model
from typing import NamedTuple, List

class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: tf.Tensor # Dict[Action, float]
    hidden_state: List[float] # not sure about this one lol

class Network_FC(object):
    def __init__(self, s_in=4, config):
        # Initialise a uniform network - should I init these networks explicitly?
        action_space_size = config.action_space_size
        self.f = PredNet_FC(s_in, n_acts)
        self.g = DynaNet_FC((5,5,nf+1), nf)
        self.h = ReprNet_FC((80,80,32), nf)
        self.steps = 0

    # TODO: Think about what dtypes we want to return/save
    # state should be Tensor because it is re-used in recurrent_inference()
    # Should the value,reward scalars be stored as floats?
    # Should action_logits/policy be stored as np.array or tf.Tensor?
    def initial_inference(self, obs) -> NetworkOutput:
        # representation + prediction function
        # input: 32x80x80 observation
        state = self.h(obs)
        policy_logits, value = self.f(state)
        policy = policy_logits[0]
        return NetworkOutput(float(value[0]), 0.0, policy.numpy(), state) # keep state 4D
    
    def recurrent_inference(self, state, action) -> NetworkOutput:
        # dynamics + prediction function
        # Input: hidden state nfx5x5
        # Concat/pad action to channel dim of states
        state_action = tf.pad(state, paddings=[[0, 0], [0, 0], [0, 0], [0, 1]], constant_values=action)
        next_state, reward =  self.g(state_action)
        policy_logits, value = self.f(next_state)
        policy = policy_logits[0]
        return NetworkOutput(float(value[0]), reward[0], policy.numpy(), next_state)

    def get_weights(self):
        # Returns the weights of this network.
        self.steps += 1 # probably not ideal
        return [self.f.parameters(), self.g.parameters(), self.h.parameters()]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
    
### FC Tensorflow model definitions ###

def ReprNet_FC(input_shape, nf=128):
    o = Input(shape=input_shape)
    x = ConvBlock(o, nf, 2)
    x = ConvBlock(x, nf, 2)
    x = ConvBlock(x, nf, 2)
    s = ConvBlock(x, nf, 2)
    return Model(o, s)

def DynaNet_FC(input_shape, nf=128):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf)
    x = ConvBlock(x, nf)
    s_new = ConvBlock(x, nf)
    
    r = Flatten()(s_new)
    r = Dense(1)(r)
    return Model(s, [s_new, r])

def PredNet_FC(input_shape, nf=64, num_actions=4):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf//2) 
    x = Flatten()(x)

    a = Dense(num_actions)(x)
    v = Dense(1)(x)
    return Model(s, [a, v])
    
        
class Network_CNN(object):
    def __init__(self, h_in=5, w_in=5, c_in=32, nf=128, n_acts=4):
        # Initialise a uniform network - should I init these networks explicitly?
        self.f = PredNet_CNN((5,5,nf), nf, n_acts)
        self.g = DynaNet_CNN((5,5,nf+1), nf)
        self.h = ReprNet_CNN((80,80,32), nf)
        self.steps = 0

    # TODO: Think about what dtypes we want to return/save
    # state should be Tensor because it is re-used in recurrent_inference()
    # Should the value,reward scalars be stored as floats?
    # Should action_logits/policy be stored as np.array or tf.Tensor?
    def initial_inference(self, obs) -> NetworkOutput:
        # representation + prediction function
        # input: 32x80x80 observation
        state = self.h(obs)
        policy_logits, value = self.f(state)
        policy = policy_logits[0]
        return NetworkOutput(float(value[0]), 0.0, policy.numpy(), state) # keep state 4D
    
    def recurrent_inference(self, state, action) -> NetworkOutput:
        # dynamics + prediction function
        # Input: hidden state nfx5x5
        # Concat/pad action to channel dim of states
        state_action = tf.pad(state, paddings=[[0, 0], [0, 0], [0, 0], [0, 1]], constant_values=action)
        next_state, reward =  self.g(state_action)
        policy_logits, value = self.f(next_state)
        policy = policy_logits[0]
        return NetworkOutput(float(value[0]), reward[0], policy.numpy(), next_state)

    def get_weights(self):
        # Returns the weights of this network.
        self.steps += 1 # probably not ideal
        return [self.f.parameters(), self.g.parameters(), self.h.parameters()]

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
        
    
### CNN Tensorflow model definitions ###

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

def ReprNet_CNN(input_shape=(80,80,32), nf=128):
    o = Input(shape=input_shape)
    x = ConvBlock(o, nf, 2)
    x = ConvBlock(x, nf, 2)
    x = ConvBlock(x, nf, 2)
    s = ConvBlock(x, nf, 2)
    return Model(o, s)

def DynaNet_CNN(input_shape=(5,5,129), nf=128):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf)
    x = ConvBlock(x, nf)
    s_new = ConvBlock(x, nf)
    
    r = Flatten()(s_new)
    r = Dense(1)(r)
    return Model(s, [s_new, r])

def PredNet_CNN(input_shape=(5,5,128), nf=64, num_actions=4):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf//2) 
    x = Flatten()(x)

    a = Dense(num_actions)(x)
    v = Dense(1)(x)
    return Model(s, [a, v])
