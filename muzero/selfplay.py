import ray
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pdb
import time

import matplotlib
#matplotlib.use("MacOSX")
from matplotlib import pyplot as plt

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .models import Network, Network_FC, Network_CNN
from .env import Game
from .mcts_numpy import Node, expand_node, run_mcts, add_exploration_noise, select_action

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
@ray.remote
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    for i in tqdm(range(config.selfplay_iterations), desc='Self-play iter'):
        network = Network_FC(config) if config.model_type == "fc" else Network_CNN(config)
        network_weights = ray.get(storage.latest_weights.remote()) # serial/blocking call
        network.set_weights(network_weights)
        #         network = storage.latest_network()

        game,_ = play_game(config, network)
        game.prepare_to_save()
        replay_buffer.save_game.remote(game) # should we use ray.put() here??


    
### Run 1 Game/Trajectory ###
    
# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network, render=False) -> Game:
    game = Game(config)

    total_reward = 0
    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function h to
        # obtain a hidden state given the current observation.
        root = Node(config, action=None)
        current_observation = tf.expand_dims(game.make_image(-1), 0) # 1x80x80x32 tf.Tensor - needs dummy batch dim
        expand_node(config, root, game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        reward = game.apply(action)
        total_reward += reward
        game.store_search_statistics(root)
        
        if render:
            game.env.gym_env.render(mode='human')
            #time.sleep(.1)
    if render:
        print(f"Game Reward: {total_reward}")

    return game, total_reward
