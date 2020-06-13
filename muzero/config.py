from typing import Optional

from .common import KnownBounds

class MuZeroConfig(object):

    def __init__(self,
                gym_env_name: str,
                action_space_size: int,
                value_support_size: int,
                reward_support_size: int,
                selfplay_iterations: int,
                max_moves: int,
                discount: float,
                use_TD_values: bool,
                dirichlet_alpha: float,
                num_simulations: int,
                batch_size: int,
                td_steps: int,
                num_actors: int,
                lr_init: float,
                lr_decay_steps: float,
                checkpoint_interval: float,
                visit_softmax_temperature_fn,
                known_bounds: Optional[KnownBounds] = None,
                num_train_gpus = 0):
        # Env
        self.gym_env_name = gym_env_name
        
        # Model
        self.model_type = "fc" # "cnn"
        
        ### Self-Play
        self.action_space_size = action_space_size
        self.value_support_size = value_support_size
        self.reward_support_size = reward_support_size
        self.num_actors = num_actors


        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount
        self.use_TD_values = use_TD_values

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.selfplay_iterations = int(selfplay_iterations) ##
        self.training_steps = int(1e6)
        self.checkpoint_interval = int(checkpoint_interval) #1e3
        self.window_size = int(1e2)
        self.batch_size = batch_size
        self.num_unroll_steps = 5 # 5
        self.td_steps = td_steps

        # PER
        self.PER_alpha = 1.0 # alpha = 0 is uniform sampling
        self.PER_beta = 1.0

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        # Saving
        self.checkpoint_interval = checkpoint_interval
        # Training devices
        self.num_train_gpus = num_train_gpus