import random
import ray
import numpy as np
import tensorflow as tf
import pdb
from _collections import deque

from .config import MuZeroConfig
from .env import Game
from .models import Network, Network_CNN, Network_FC

@ray.remote
class ReplayBuffer(object):
    """Stores the target tuples to sample from later during training"""
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = deque(maxlen=config.window_size)
        # self.game_priorities = deque(maxlen=config.window_size)
        self.config = config

    def save_game(self, game):
        # Pop off oldest replays off the left of the deque to make space for new ones automatically
        self.buffer.append(game)
        # self.update_priorities(game)

    def calculate_game_priority(self, game):
        """
        Can choose mean, sum, max over positional priorities to proxy for game priority
        Todo: Adapt for SumTree for sub-linear performance?
        """
        return np.max(game.position_priorities)

    def update_priorities(self, game):
        gp = self.calculate_game_priority(game)
        self.game_priorities.append(gp)

    def sample_batch(self, K: int, td: int, network_weights):
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
        policy_masks = []
        IS_weightings = []
        observations_all = []
        
        # Sample a batch size worth of games
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, gp, *self.sample_position(g)) for g, gp in games]
        
        for g, gp, i, pp in game_pos:
            observations.append(g.make_image(i))
            # Todo: this needs to be td steps aheadBut
            observations_all.append([g.make_image(t) for t in range(i,i+K+1)])
            # Use -1 padding for actions, this should get masked anyway
            action_history = g.history[i:i + K]
            # Game and position IS weightings
            game_weighting = np.ones(K+1)/gp
            position_weighting = np.pad(pp[i:i+K+1], (0, (K+1)-len(pp[i:i+K+1])), constant_values=(1,0)) # Todo: how to weight absorbing states?
            IS_weighting = (game_weighting*position_weighting)**self.config.PER_beta

            def random_pad(vector, pad_width, iaxis, kwargs):
                vector[:pad_width[0]] = -1 # np.random.randint(20, 30, size=pad_width[0])
                vector[vector.size-pad_width[1]:] = np.random.choice(vector[pad_width[0]:], size=pad_width[1])

            action_history_padded = np.pad(action_history, (1, K-len(action_history)), mode=random_pad)
            actions.append(action_history_padded)

            z,u,pi,mask,policy_mask = g.make_target(i, K, td)
            # z,u,pi,mask,policy_mask = g.make_target.remote(g, i, K, td, network_weights)
            target_values.append(z)
            target_rewards.append(u)
            target_policies.append(pi)
            masks.append(mask)
            policy_masks.append(policy_mask)
            IS_weightings.append(IS_weighting)

        network = Network_FC(self.config) if self.config.model_type == "fc" else Network_CNN(self.config)
        network.set_weights(network_weights)
        # value = self.compute_target_value(current_index, td, network)
        target_values = []
        for i in range(K+1):
            obs = tf.stack([o[i] for o in observations_all], axis=0)
            network_output = network.initial_inference(obs, convert_to_scalar=True)
            target_values.append(network_output * 0.997 ** td)

        return (
                tf.stack(observations, axis=0),
                tf.cast(tf.stack(actions, axis=0), dtype=tf.int32),
                # tf.expand_dims(tf.cast(tf.stack(target_values, axis=0),dtype=tf.float32),axis=-1),
                tf.expand_dims(tf.cast(tf.stack(target_values, axis=1), dtype=tf.float32), axis=-1),
                tf.expand_dims(tf.stack(target_rewards, axis=0),axis=-1),
                tf.stack(target_policies, axis=0),
                tf.expand_dims(tf.cast(tf.stack(masks, axis=0),dtype=tf.float32),axis=-1),
                tf.expand_dims(tf.cast(tf.stack(policy_masks, axis=0), dtype=tf.float32), axis = -1),
                tf.cast(tf.expand_dims(tf.stack(IS_weightings, axis=0), axis=-1), dtype=tf.float32)
               )

    def sample_game(self) -> Game:
        # TODO: figure out sampling regime
        # Todo: Add PER
        # Todo: Add IS weighting
        # Sample game from buffer either uniformly or according to some priority e.g. importance sampling.
        # p = np.array(self.game_priorities)
        # p /= p.sum()
        # game_ix = np.random.choice(len(self.buffer), p=p)
        game_ix = random.randint(0,len(self.buffer)-1) # random uniform
        return self.buffer[game_ix], self.get_buffer_size()

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        p = np.array(game.position_priorities)
        p /= p.sum()
        pos_ix = np.random.choice(len(game.root_values), p=p)
        # pos_ix = random.randint(0,len(game.root_values)-1) # random uniform
        return pos_ix, p
    
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