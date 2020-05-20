##################################
####### Part 2: Training #########

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.compat.v1.train import get_global_step
import ray
from tqdm import tqdm
import time

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .models import Network

cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def batch_to_tf_batch(batch):
    image_batch = tf.stack([tf.squeeze(b[0]) for b in batch],axis=0)
    action_batch = tf.stack([tf.pad(b[1],paddings=[[0,5-len(b[1])]]) for b in batch], axis=0)
    # This target one won't work because the (v,r,a) tuple is <6 elements when a is []
    target_batch = tf.stack([tf.pad(tf.stack([(v,r,*a) for v,r,a in b[2]]), paddings=[[0,5+1-len(b[2])],[0,0]]) for b in batch])
    return (image_batch, action_batch, reward_batch)

# @ray.remote
def train_network(config: MuZeroConfig, storage: SharedStorage,
                                    replay_buffer: ReplayBuffer):
#     while len(replay_buffer.buffer)==0: time.sleep(1)
    while ray.get(replay_buffer.get_buffer_size.remote()) < 1:
        time.sleep(1)
    network = Network()
    learning_rate = config.lr_init * config.lr_decay_rate**(
#           get_global_step() / config.lr_decay_steps)
            1)
    optimizer = Adam(learning_rate, config.momentum)

    for i in tqdm(range(config.training_steps), desc='Training iter'):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        ray_id = replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps)
        batch = ray.get(ray_id)
        vector_batch = batch_to_tf_batch(batch)
        train_step(optimizer, network, vector_batch, config.weight_decay)
#         update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


# BPTT - this code can surely be vectorised - yes it really does rather than looping over batches
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
                scalar_loss(value, target_value) + # value
                scalar_loss(reward, target_reward) + # reward
                cce_loss_logits(policy_logits, target_policy) # action
            )

            loss += scale_gradient(l, gradient_scale)

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)
        
        
# @tf.function
def train_step(optimizer: Optimizer, network: Network, batch,
                                     weight_decay: float):
    """
    Batch is 3-tuple of:
    Image (N,80,80,1)
    Actions (N,K)
    Targets (N,K+1,(v,p,r))
    """
    image, actions, targets = batch
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).

        K = targets.shape[1] # seqlen
        for k in range(K):
            if k==0:
                # Initial step, from the real observation.
                value, reward, policy_logits, hidden_state = network.initial_inference(image)
                gradient_scale, value, reward, policy_logits = (1.0, value, reward, policy_logits)
            else:
                # All following steps
                value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action[:,k])
                gradient_scale, value, reward, policy_logits = (1.0 / len(actions), value, reward, policy_logits)

            hidden_state = scale_gradient(hidden_state, 0.5)
            
            target_value, target_reward, target_policy = target[:,k,:]

            l = (
                cce_loss(value, target_value) + # value
                cce_loss(reward, target_reward) + # reward
                cce_loss_logits(policy_logits, target_policy) # action
            )

            loss += scale_gradient(l, gradient_scale)
        
        # Todo: Eventually we want to use keras layer regularization or AdamW
        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)
            
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)
#     train_accuracy(labels, predictions)


# Use categorical/softmax cross-entropy loss rather than binary/logistic
# Value and reward are non-logits, actions are logits
def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return cce_loss(prediction, target)