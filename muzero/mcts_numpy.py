from typing import Optional, List
import tensorflow as tf
import math
import numpy as np
import collections

from .common import MAXIMUM_FLOAT_VALUE, KnownBounds
from .config import MuZeroConfig
from .models import Network, NetworkOutput

# MCTS optimised for Numpy
# Inspiration: https://www.moderndescartes.com/essays/deep_dive_mcts/

class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    
    def normalize(self, value: np.array) -> np.array:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
    
class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_value_sum = collections.defaultdict(float)
        self.child_visit_count = collections.defaultdict(int)
        self.child_rewards = collections.defaultdict(float)

# Change so that each node knows children stats rather than its own
class Node(object):

    def __init__(self, config: MuZeroConfig, action: int, parent = DummyNode()):
        self.children = {} # can change to list is we change code elsewhere to use .append
        self.hidden_state = None
        
        self.parent = parent
        self.action = action
        self.is_expanded = False
        self.child_priors = np.zeros([config.action_space_size], dtype=np.float32)
        self.child_value_sum = np.zeros([config.action_space_size], dtype=np.float32)
        self.child_visit_count = np.zeros([config.action_space_size], dtype=np.int32) # using floats so arithmetic works
        self.child_rewards = np.zeros([config.action_space_size], dtype=np.float32)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def child_values(self) -> np.array:
        # This should avoid div by 0 errors
        return np.divide(self.child_value_sum, self.child_visit_count, 
                         out=np.zeros_like(self.child_value_sum), where=self.child_visit_count!=0)
    
    # These are proxies for the visit count, sum and reward values - we grab them from the parent
    @property
    def visit_count(self):
        return self.parent.child_visit_count[self.action]

    @visit_count.setter
    def visit_count(self, value):
        self.parent.child_visit_count[self.action] = value

    @property
    def value_sum(self):
        return self.parent.child_value_sum[self.action]

    @value_sum.setter
    def value_sum(self, value):
        self.parent.child_value_sum[self.action] = value
        
    @property
    def reward(self):
        return self.parent.child_rewards[self.action]

    @reward.setter
    def reward(self, value):
        self.parent.child_rewards[self.action] = value

    
class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[int], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: int):
        self.history.append(action)

    def last_action(self) -> int:
        return self.history[-1]

    def action_space(self) -> List[int]:
        return [i for i in range(self.action_space_size)]
    
    
# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory,
             network: Network):
    """TODO: Multithread"""
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        # Traverse tree, expanding by highest UCB until leaf reached
        while node.is_expanded:
            node.visit_count += 1 # This is part of the virtual losses trick
            action = best_move(config, node, min_max_stats) # UCB selection
            node = maybe_add_child(config, action, node) # adds child if it doesn't already exist
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2] # parent of leaf
        # Dynamics: g(s_{k-1},a_k) = r_k, s_k
        # Predictions: f(s_k) = v_k, p_k
        # -> (v,r,p,s)
        network_output = network.recurrent_inference(parent.hidden_state,
                                                     tf.expand_dims(history.last_action(),0)) # Needs batch dim and to be float
        # expand node using v,r,p predictions from NN
        expand_node(config, node, history.action_space(), network_output)

        # back up values to the root node
        backpropagate(search_path, float(network_output.value[0]), config.discount, 
                      min_max_stats)

        
### i. Selection: UCB Child Selection ###
        
# Select the child with the highest UCB score.
def best_move(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    action = np.argmax(ucb_score(config, node, min_max_stats)) # ucb_score should return a np.array
    return int(action)

def maybe_add_child(config: MuZeroConfig, action: int, node: Node):
    if action not in node.children:
        node.children[action] = Node(config, action, node)
    return node.children[action]

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
# UCB score here should be across all children not just 1 child
def ucb_score(config: MuZeroConfig, parent: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = np.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (parent.child_visit_count + 1)

    # P(s,a)*pb_c
    prior_score = pb_c * parent.child_priors
    # Q(s,a)
    value_score = parent.child_rewards + config.discount * min_max_stats.normalize(parent.child_values())
    value_score[parent.child_visit_count==0] = 0
    return prior_score + value_score


### ii. Expansion: Leaf Node Expansion + Prediction ###

# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(config: MuZeroConfig, node: Node, actions: List[int], network_output: NetworkOutput):
    """Updates predictions for state s, reward r and policy p for node based on NN outputs"""
    # Update leaf with predictions from parent
    node.is_expanded = True
    node.hidden_state = network_output.hidden_state # s
    node.reward = np.squeeze(network_output.reward) # r
    # This can be optimised
#     policy = [tf.math.exp(network_output.policy_logits[a]) for a in actions] # unnormalised probabilities
#     policy_sum = tf.reduce_sum(policy) 
    policy = np.exp(network_output.policy_logits) # unnormalised probabilities
    policy_sum = np.sum(policy) 
    node.child_priors = policy / policy_sum
        

#### iii. Backup: Search Tree Update/Backprop

# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats):
    # Traverse back up UCB search path
    for node in reversed(search_path):
        node.value_sum += value # if node.to_play == to_play else -value
#         node.visit_count += 1
        min_max_stats.update(node.value())
        value = discount * value + node.reward
        
        
### Exploration Noise ###

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: MuZeroConfig, node: Node):
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size)
    frac = config.root_exploration_fraction
    node.child_priors = node.child_priors * (1 - frac) + noise * frac

        
### Softmax search policy $\pi$ ###

def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network, greedy_policy=False) -> int:
    """Search policy: softmax probability over actions dictated by visited counts"""
    # Visit counts of children nodes - policy proportional to counts
    if greedy_policy:
        action = np.argmax(node.child_visit_count)
    else:
        # Get softmax temp
        t = config.visit_softmax_temperature_fn(
            num_moves=num_moves, training_steps=network.training_steps())
        action = softmax_sample(node.child_visit_count, t)
    return action

def softmax_sample(distribution, T: float):
    counts = distribution
    actions = range(len(counts))
    softmax_probs = softmax(counts/T) # tf.nn.softmax(counts/T, axis=None)
    sampled_action = np.random.choice(actions, size=1, p=softmax_probs)[0]
    return sampled_action

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)