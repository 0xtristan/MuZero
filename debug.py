from muzero.config import MuZeroConfig
from muzero.main import Muzero
from muzero.common import KnownBounds
import ray

def make_atari_config() -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 5e3:
            return 1.0
        elif training_steps < 1e4:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        gym_env_name='CartPole-v1',
        action_space_size=2,
        value_support_size=10,
        reward_support_size=10,
        selfplay_iterations=1000, # Todo: implement None for continuous play
        max_moves=500,
        discount=0.997,
        use_TD_values=True,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=128,#1024,
        td_steps=50,#10
        num_actors=1,#350
        lr_init=0.05,#0.05
        lr_decay_steps=350e3,
        checkpoint_interval=5,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        # known_bounds=KnownBounds(min=0, max=500),
        num_train_gpus=0)

ray.init(local_mode=False)

config = make_atari_config()
mz = Muzero(config)

mz.run()