import ray
from tqdm import tqdm
import numpy as np

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .models import Network
from .env import Game, Node
from .mcts import expand_node, run_mcts

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
@ray.remote
def run_selfplay(config: MuZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    # tf.summary.trace_on()
    for i in tqdm(range(config.selfplay_iterations), desc='Self-play iter'):
        network = storage.latest_network()
        game = play_game(config, network)
        print(game.root_values)
        replay_buffer.save_game(game)
        # tf.summary.trace_export("Selfplay", step=i, profiler_outdir='logs')
    # tf.summary.trace_off()

    
### Run 1 Game/Trajectory ###
    
# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = Game(config.action_space_size, config.discount)

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function h to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1) # 80x80x32
        expand_node(root, game.legal_actions(),
                    network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


### Exploration Noise ###

# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
# @tf.function
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

        
### Softmax search policy $\pi$ ###
        
# @tf.function
def select_action(config: MuZeroConfig, num_moves: int, node: Node,
                  network: Network) -> int:
    """Search policy: softmax probability over actions dictated by visited counts"""
    # Visit counts of chilren nodes - policy proportional to counts
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    # Get softmax temp
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps())
    action = softmax_sample(visit_counts, t)
    return action

# @tf.function
def softmax_sample(distribution, T: float):
    counts = np.array([d[0] for d in distribution])
    actions = [d[1] for d in distribution]
    softmax_probs = softmax(counts/T)
    sampled_action = np.random.choice(actions, size=1, p=softmax_probs)[0]
    return sampled_action
#     return nn.Softmax(dim=0)(distribution/T)

# @tf.function
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)