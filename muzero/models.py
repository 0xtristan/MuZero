import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Add, ReLU, Input, Flatten, LeakyReLU
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow_core.python.keras import regularizers
from typing import NamedTuple, List, Callable
from abc import ABC, abstractmethod
import ray

class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: tf.Tensor # Dict[Action, float]
    hidden_state: List[float] # not sure about this one lol

# This defines the abstract base class for a MuZero network
# Inference methods must be defined in classes implementing this base class
class Network(ABC):
    def __init__(self):
        # Initialise a uniform network - should I init these networks explicitly?
        super().__init__()
        self.f = None
        self.g = None
        self.h = None
        self.steps = 0
    
    @abstractmethod
    def initial_inference(self, obs) -> NetworkOutput:
        raise NotImplementedError
    
    @abstractmethod
    def recurrent_inference(self, state, action) -> NetworkOutput:
        raise NotImplementedError

    def get_weights(self):
        """Retrieves weight tensors
        Todo: In future come up with a good way to save load from disk - probs just model.save_weights() """
        # Returns the weights of this network.
#         self.steps += 1 # probably not ideal
        return (self.f.get_weights(), self.g.get_weights(), self.h.get_weights())
   
    # Todo: potentially include remote weight setting here - would mean networks would need access to storage worker
    def set_weights(self, weights):
        f_w, g_w, h_w = weights
        self.f.set_weights(f_w)
        self.g.set_weights(g_w)
        self.h.set_weights(h_w)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return self.steps
    
    
class Network_FC(Network):
    def __init__(self, config, s_in=4, h_size=32, repr_size=4):
        super().__init__()
        # Initialise a uniform network - should I init these networks explicitly?
        n_acts = config.action_space_size
        self.value_support_size = config.value_support_size
        self.reward_support_size = config.reward_support_size
        self.regularizer = regularizers.l2(config.weight_decay)
        self.f = PredNet_FC((repr_size,), n_acts, h_size, support_size=self.value_support_size, regularizer=self.regularizer)
        # self.fv = PredNetV_FC((h_size,), h_size, support_size=self.value_support_size, regularizer=self.regularizer)
        # self.fa = PredNetA_FC((h_size,), n_acts, h_size, regularizer=self.regularizer)
        self.g = DynaNet_FC((repr_size+config.action_space_size,), repr_size, h_size, support_size=self.reward_support_size, regularizer=self.regularizer)
        self.h = ReprNet_FC((s_in,), repr_size, h_size, regularizer=self.regularizer)
        self.steps = 0
        self.config = config

    def initial_inference(self, obs, convert_to_scalar = True) -> NetworkOutput:
        # representation + prediction function
        # input: 32x80x80 observation for breakout
        state = self.h(obs)
        policy_logits, value = self.f(state)
        # policy_logits, value = self.fa(state), self.fv(state)
        reward = tf.ones((self.config.batch_size,1)) # Todo: test this
        if convert_to_scalar:
            value = support_to_scalar(value, self.value_support_size)
            # reward = support_to_scalar(reward, self.reward_support_size)
        else:
            reward = scalar_to_support(reward, self.reward_support_size)

        return NetworkOutput(value, reward, policy_logits, state)
    
    def recurrent_inference(self, state, action, convert_to_scalar = True) -> NetworkOutput:
        # dynamics + prediction function
        # Input: hidden state nfx5x5
        # Concat/pad action to channel dim of states
        action_ohe = tf.one_hot(action, self.config.action_space_size)
        state_action = tf.concat([state,action_ohe], axis=1)
        next_state, reward = self.g(state_action)
        policy_logits, value = self.f(next_state)
        # policy_logits, value = self.fa(next_state), self.fv(next_state)
        if convert_to_scalar:
            value = support_to_scalar(value, self.value_support_size)
            reward = support_to_scalar(reward, self.reward_support_size)
        return NetworkOutput(value, reward, policy_logits, next_state)

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.f, self.g, self.h)
            return [variables
                    for variables_list in map(lambda n: n.weights, networks)
                    for variables in variables_list]

        return get_variables
    
### FC Tensorflow model definitions ###

def ReprNet_FC(input_shape, repr_size, h_size, regularizer):
    o = Input(shape=input_shape)
    x = o
    x = Dense(repr_size, kernel_regularizer=regularizer)(x)
    x = ReLU()(x)
    s = Dense(repr_size, kernel_regularizer=regularizer)(x) # Since we have +ve and -ve positions, angles, velocities
    s = min_max_scaling(s) # This replaces our activation fn
    return Model(o, s)

def DynaNet_FC(input_shape, repr_size, h_size, support_size, regularizer):
    s = Input(shape=input_shape)
    x = s

    s_new = Dense(h_size, kernel_regularizer=regularizer)(x)
    s_new = ReLU()(s_new)
    # s_new = Dense(h_size, kernel_regularizer=regularizer)(s_new)
    # s_new = ReLU()(s_new)

    r = Dense(h_size, kernel_regularizer=regularizer)(x)
    r = ReLU()(r)
    # r = Dense(h_size, kernel_regularizer=regularizer)(r)
    # r = ReLU()(r)
    
    s_new = Dense(repr_size, kernel_regularizer=regularizer)(s_new)
    s_new = min_max_scaling(s_new)
    # r = LeakyReLU()(s_new)
    r = Dense(support_size*2+1, kernel_regularizer=regularizer)(r) # rewards are 1 for each frame it stays upright, 0 otherwise
    return Model(s, [s_new, r])

def PredNet_FC(input_shape, num_actions, h_size, support_size, regularizer):
    s = Input(shape=input_shape)
    x = s

    a = Dense(h_size, kernel_regularizer=regularizer)(x)
    a = ReLU()(a)
    # a = Dense(h_size, kernel_regularizer=regularizer)(a)
    # a = ReLU()(a)

    v = Dense(h_size, kernel_regularizer=regularizer)(x)
    v = ReLU()(v)
    # v = Dense(h_size, kernel_regularizer=regularizer)(v)
    # v = ReLU()(v)
    
    a = Dense(num_actions, kernel_regularizer=regularizer)(a) # policy should be logits
    v = Dense(support_size*2+1, kernel_regularizer=regularizer)(v) # This can be a large number
    return Model(s, [a, v])

# def PredNetV_FC(input_shape, h_size, support_size, regularizer):
#     s = Input(shape=input_shape)
#     x = Dense(h_size, kernel_regularizer=regularizer)(s)
#     x = ReLU()(x)
# #     x = Dense(h_size, kernel_regularizer=regularizer)(x)
# #     x = ReLU()(x)
#     v = Dense(support_size*2+1, kernel_regularizer=regularizer)(x) # This can be a large number
#     return Model(s, v)
#
# def PredNetA_FC(input_shape, num_actions, h_size, regularizer):
#     s = Input(shape=input_shape)
#     x = Dense(h_size, kernel_regularizer=regularizer)(s)
#     x = LeakyReLU()(x)
# #     x = Dense(h_size, kernel_regularizer=regularizer)(x)
# #     x = LeakyReLU()(x)
#     a = Dense(num_actions, kernel_regularizer=regularizer)(x) # policy should be logits
#     return Model(s, a)

def min_max_scaling(tensor, eps = 1e-12):
    """ Rescales tensor linearly to range [0,1]. See appendix G of paper """
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    return (tensor - min_val + eps) / tf.maximum((max_val - min_val), 2 * eps)

def support_to_scalar(logits, support_size, eps = 0.001):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = tf.nn.softmax(logits, axis=1)
    support = tf.expand_dims(tf.range(-support_size, support_size + 1), axis=0)
    support = tf.tile(support, [logits.shape[0], 1])  # make batchsize supports
    # Expectation under softmax
    x = tf.cast(support, tf.float32) * probabilities
    x = tf.reduce_sum(x, axis=-1)
    # Inverse transform h^-1(x) from Lemma A.2.
    # From "Observe and Look Further: Achieving Consistent Performance on Atari" - Pohlen et al.
    x = tf.math.sign(x) * (((tf.math.sqrt(1.+4.*eps*(tf.math.abs(x)+1+eps))-1)/(2*eps))**2-1)
    x = tf.expand_dims(x, 1)
    return x

def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = tf.math.sign(x) * (tf.math.sqrt(tf.math.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    # input (N,1)
    x = tf.clip_by_value(x, -support_size, support_size) # 50.3
    floor = tf.math.floor(x) # 50
    prob_upper = x - floor # 0.3
    prob_lower = 1 - prob_upper # 0.7
    # Needs to become (N,601)
    dim1_indices = tf.cast(tf.math.floor(x)+support_size, tf.int32)
    dim0_indices = tf.expand_dims(tf.range(0,x.shape[0]), axis=1) # this is just 0,1,2,3
    lower_indices = tf.concat([dim0_indices, dim1_indices], axis=1)

    supports = tf.scatter_nd(lower_indices, tf.squeeze(prob_lower, axis=1), shape=(x.shape[0],2*support_size+1))
    higher_indices = tf.concat([dim0_indices, tf.clip_by_value(dim1_indices+1,0,2*support_size)], axis=1)
    supports = tf.tensor_scatter_nd_add(supports, higher_indices, tf.squeeze(prob_upper, axis=1))
    return supports
    
        
class Network_CNN(Network):
    def __init__(self, h_in=5, w_in=5, c_in=32, nf=128, n_acts=4):
        super().__init__()
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
    # Todo: Input normalisation (esp for images)
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf)
    x = ConvBlock(x, nf)
    s_new = ConvBlock(x, nf, activation='sigmoid')
    
    r = Flatten()(s_new)
    r = Dense(1)(r) # Rewards can usually scale arbitrarily high - needs support implementation
    return Model(s, [s_new, r])

def PredNet_CNN(input_shape=(5,5,128), nf=64, num_actions=4):
    s = Input(shape=input_shape)
    x = ConvBlock(s, nf)
    x = ConvBlock(x, nf//2) 
    x = Flatten()(x)

    a = Dense(num_actions)(x) # policy logits - no activation
    v = Dense(1)(x) # value probably has to be >1 from unscaled rewards
    return Model(s, [a, v])
