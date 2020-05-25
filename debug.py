import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices())

from muzero.config import MuZeroConfig
from muzero.main import Muzero

import ray
import gym

e = gym.make('CartPole-v0')

print(e.observation_space) # Cart Position, Cart Velocity, Pole Angle, Velocity at Tip
print(e.action_space) # Push Cart Left/Right

gpu_count = len(tf.config.experimental.list_physical_devices('GPU'))

def make_cartpole_config() -> MuZeroConfig:

    def visit_softmax_temperature(num_moves, training_steps):
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        gym_env_name='CartPole-v0',
        action_space_size=2,
        selfplay_iterations=1, # Todo: implement None for continuous play
        max_moves=27,#000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        batch_size=10,#24,#1024,
        td_steps=10,
        num_actors=1,#350,
        lr_init=0.05,
        lr_decay_steps=350e3,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        num_train_gpus=gpu_count)


ray.init()
config = make_cartpole_config()
mz = Muzero(config)

mz.run()