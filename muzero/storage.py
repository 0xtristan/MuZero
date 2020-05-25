import random
import ray
import numpy as np
import tensorflow as tf
import pdb

from .config import MuZeroConfig
from .env import Game
from .models import Network, Network_CNN, Network_FC

@ray.remote
class ReplayBuffer(object):
    """Stores the target tuples to sample from later during training"""
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.config = config

    def save_game(self, game):
        # Pop off oldest replays to make space for new ones
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, K: int, td: int):
        """
        Inputs
            K: num unroll steps
            td: num TD steps
        Outputs
            (observation, next K actions, target tuple)
        """
        observations = []
        actions = []
        target_values = []
        target_rewards = []
        target_policies = []
        masks = []
        
        # Sample a batch size worth of games
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        
        for g, i in game_pos:
            observations.append(g.make_image(i))
            # Use -1 padding for actions, this should get masked anyway
            action_history = g.history[i:i + K]
            action_history_padded = np.pad(action_history, (1, K-len(action_history)), 
                                           constant_values=(-1,-1)).astype('float32')
            actions.append(action_history_padded)
            
            z,u,pi,mask = g.make_target(i, K, td)
            target_values.append(z)
            target_rewards.append(u)
            target_policies.append(pi)
            masks.append(mask)
        
        return (
                tf.stack(observations, axis=0),
                tf.stack(actions, axis=0),
                tf.expand_dims(tf.cast(tf.stack(target_values, axis=0),dtype=tf.float32),axis=-1),
                tf.expand_dims(tf.stack(target_rewards, axis=0),axis=-1),
                tf.stack(target_policies, axis=0),
                tf.expand_dims(tf.cast(tf.stack(masks, axis=0),dtype=tf.float32),axis=-1)
               )

    def sample_game(self) -> Game:
        # TODO: figure out sampling regime
        # Sample game from buffer either uniformly or according to some priority e.g. importance sampling.
        game_ix = random.randint(0,len(self.buffer)-1) # random uniform
        return self.buffer[game_ix]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        pos_ix = random.randint(0,len(game.root_values)-1) # random uniform
        return pos_ix
    
    def get_buffer_size(self):
        return len(self.buffer)

# Needs to be rewritten so it passes weights only
# Ray can't serialise tensorflow models
@ray.remote  
class SharedStorage(object):

    def __init__(self, config):
        self._weights = {}
        self.config = config

    def latest_weights(self) -> Network:
        if self._weights:
            return self._weights[max(self._weights.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(self.config)

    def save_weights(self, step: int, weights):
        self._weights[step] = weights

def make_uniform_network(config: MuZeroConfig):
    # Todo: this is a bit shit, how can we do it better?
    net = Network_FC(config) if config.model_type == "fc" else Network_CNN(config)
    return net.get_weights()