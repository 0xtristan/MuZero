import random
import ray

from .config import MuZeroConfig
from .env import Game
from .models import Network

@ray.remote
class ReplayBuffer(object):
    """Stores the target tuples to sample from later during training"""
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

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
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + K],
                 g.make_target(i, K, td))
                for (g, i) in game_pos]

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
    
# @ray.remote  
class SharedStorage(object):

    def __init__(self, config):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network(config)

    def save_network(self, step: int, network: Network):
        self._networks[step] = network

def make_uniform_network(config: MuZeroConfig):
    return Network_FC(config) if self.model_type = "fc" else Network_CNN(config)