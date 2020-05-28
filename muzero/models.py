import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Add, ReLU, Input, Flatten
from tensorflow.keras.models import Model
from typing import NamedTuple, List
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
        self.steps += 1 # probably not ideal
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
    def __init__(self, config, s_in=4, h_size=16):
        super().__init__()
        # Initialise a uniform network - should I init these networks explicitly?
        n_acts = config.action_space_size
        self.f = PredNet_FC((h_size,), n_acts, h_size, config.value_support_size)
        self.g = DynaNet_FC((h_size+1,), h_size)
        self.h = ReprNet_FC((s_in,), h_size)
        self.steps = 0

    # TODO: Think about what dtypes we want to return/save
    # state should be Tensor because it is re-used in recurrent_inference()
    # Should the value,reward scalars be stored as floats?
    # Should action_logits/policy be stored as np.array or tf.Tensor?
    def initial_inference(self, obs) -> NetworkOutput:
        # representation + prediction function
        # input: 32x80x80 observation # TODO-No?
        state = self.h(obs)
        policy_logits, value = self.f(state)
        return NetworkOutput(value, tf.zeros_like(value), policy_logits, state) # drop batch dim with [0], state still has batch
    
    def recurrent_inference(self, state, action) -> NetworkOutput:
        # dynamics + prediction function
        # Input: hidden state nfx5x5
        # Concat/pad action to channel dim of states
        state_action = tf.concat([state,tf.expand_dims(action,axis=-1)], axis=-1)
        next_state, reward =  self.g(state_action)
        policy_logits, value = self.f(next_state)
        return NetworkOutput(value, reward, policy_logits, next_state)
    
### FC Tensorflow model definitions ###

def ReprNet_FC(input_shape, h_size):
    o = Input(shape=input_shape)
    x = Dense(h_size, activation='relu')(o)
    s = Dense(h_size, activation='sigmoid')(x) # Since we have +ve and -ve positions, angles, velocities
    return Model(o, s)

def DynaNet_FC(input_shape, h_size):
    s = Input(shape=input_shape)
    x = Dense(h_size, activation='relu')(s)
    x = Dense(h_size, activation='relu')(x)
    s_new = Dense(h_size, activation='sigmoid')(x)
    r = Dense(1, activation='sigmoid')(x) # rewards are 1 for each frame it stays upright, 0 otherwise
    return Model(s, [s_new, r])

def PredNet_FC(input_shape, num_actions, h_size, support_size):
    s = Input(shape=input_shape)
    x = Dense(h_size, activation='relu')(s)
    x = Dense(h_size, activation='relu')(x)
    a = Dense(num_actions)(x) # policy should be logits
    v = Dense(support_size*2+1)(x) # This can be a large number
    v = support_to_scalar(v, support_size*2+1)
    return Model(s, [a, v])

def support_to_scalar(logits, support_size):
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
    x = tf.math.sign(x) * (((tf.math.sqrt(1+4*eps*(tf.math.abs(x)+1+eps))-1)/(2*eps))^2-1)
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
    indices = tf.concat([dim0_indices, dim1_indices], axis=1)

    supports = tf.scatter_nd(indices, tf.squeeze(prob_lower), shape=(x.shape[0],2*support_size+1))
    indices = tf.concat([dim0_indices, tf.clip_by_value(dim1_indices+1,0,2*support_size)], axis=1)
    tf.tensor_scatter_nd_add(supports, indices, tf.squeeze(prob_upper))
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
