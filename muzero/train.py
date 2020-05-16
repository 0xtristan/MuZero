##################################
####### Part 2: Training #########

from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.compat.v1.train import get_global_step
import ray

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .models import Network

@ray.remote
def train_network(config: MuZeroConfig, storage: SharedStorage,
                                    replay_buffer: ReplayBuffer):
    while len(replay_buffer.buffer)==0: pass
    network = Network()
    learning_rate = config.lr_init * config.lr_decay_rate**(
#           get_global_step() / config.lr_decay_steps)
            1)
    optimizer = Adam(learning_rate, config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def update_weights(optimizer: Optimizer, network: Network, batch,
                                     weight_decay: float):
    loss = 0
    for image, actions, targets in batch:
        # Initial step, from the real observation.
        value, reward, policy_logits, hidden_state = network.initial_inference(image)
        predictions = [(1.0, value, reward, policy_logits)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
            predictions.append((1.0 / len(actions), value, reward, policy_logits))

            hidden_state = scale_gradient(hidden_state, 0.5)

        for prediction, target in zip(predictions, targets):
            gradient_scale, value, reward, policy_logits = prediction
            target_value, target_reward, target_policy = target

            l = (
                scalar_loss(value, target_value) +
                scalar_loss(reward, target_reward) +
                tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

            loss += scale_gradient(l, gradient_scale)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return -1