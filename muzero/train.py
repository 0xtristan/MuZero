##################################
####### Part 2: Training #########

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.compat.v1.train import get_global_step
import ray
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from IPython import display

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .models import Network, Network_CNN, Network_FC

# LOSSES
cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cce_loss_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
mse = tf.keras.losses.MeanSquaredError()

# METRICS
value_loss_metric = tf.keras.metrics.Mean()
reward_loss_metric = tf.keras.metrics.Mean()
policy_loss_metric = tf.keras.metrics.Mean()
total_loss_metric = tf.keras.metrics.Mean()

# @ray.remote
def train_network(config: MuZeroConfig, storage: SharedStorage,
                                    replay_buffer: ReplayBuffer):
    while ray.get(replay_buffer.get_buffer_size.remote()) < 1:
        time.sleep(1)
    network = Network_FC(config) if config.model_type == "fc" else Network_CNN(config) # Network()
    learning_rate = config.lr_init * config.lr_decay_rate**(
#           get_global_step() / config.lr_decay_steps)
            1)
    optimizer = Adam(learning_rate, config.momentum)
    
    progbar = tf.keras.utils.Progbar(config.training_steps, verbose=1, stateful_metrics=None, unit_name='step')

    for i in range(config.training_steps): #tqdm(range(config.training_steps), desc='Training iter'):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network.get_weights())
        ray_id = replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps)
        batch = ray.get(ray_id)
        train_step(optimizer, network, batch, config.weight_decay)
#         update_weights(optimizer, network, batch, config.weight_decay)
        progbar.update(i, values=[('Value loss', value_loss_metric.result()),
                                  ('Reward loss', reward_loss_metric.result()),
                                  ('Policy loss', policy_loss_metric.result()),
                                  ('Total loss', total_loss_metric.result()),
                                 ])
        if i%10==0:
            value_loss_metric.reset_states()
            reward_loss_metric.reset_states()
            policy_loss_metric.reset_states()
            total_loss_metric.reset_states()
    storage.save_network.remote(config.training_steps, network)


def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


# # BPTT - this code can surely be vectorised - yes it really does rather than looping over batches
# def update_weights(optimizer: Optimizer, network: Network, batch,
#                                      weight_decay: float):
#     loss = 0
#     for image, actions, targets in batch:
#         # Initial step, from the real observation.
#         value, reward, policy_logits, hidden_state = network.initial_inference(image)
#         predictions = [(1.0, value, reward, policy_logits)]

#         # Recurrent steps, from action and previous hidden state.
#         for action in actions:
#             value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, action)
#             predictions.append((1.0 / len(actions), value, reward, policy_logits))

#             hidden_state = scale_gradient(hidden_state, 0.5)

#         for prediction, target in zip(predictions, targets):
#             gradient_scale, value, reward, policy_logits = prediction
#             target_value, target_reward, target_policy = target

#             l = (
#                 scalar_loss(value, target_value) + # value
#                 scalar_loss(reward, target_reward) + # reward
#                 cce_loss_logits(policy_logits, target_policy) # action
#             )

#             loss += scale_gradient(l, gradient_scale)

#     for weights in network.get_weights():
#         loss += weight_decay * tf.nn.l2_loss(weights)

#     optimizer.minimize(loss)
        
        
# @tf.function
def train_step(optimizer: Optimizer, network: Network, batch,
                                     weight_decay: float):
    """
    Batch is 3-tuple of:
    Observation (N,80,80,1) for atari or (N,4) for cartpole
    Actions (N,K+1) - k=0 is a dummy action -1
    Value Targets (N,K+1)
    Reward Targets (N,K+1)
    Policy Targets (N,K+1,A)
    Masks (N,K+1)
    """
    observations, actions, target_values, target_rewards, target_policies, masks = batch
    with tf.GradientTape() as f_tape, tf.GradientTape() as g_tape, tf.GradientTape() as h_tape:
        loss = 0
        K = actions.shape[1] # seqlen
        for k in range(K):
            if k==0:
                # Initial step, from the real observation.
                value, reward, policy_logits, hidden_state = network.initial_inference(observations)
                gradient_scale = 1.0
            else:
                # All following steps
                value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, actions[:,k])
                gradient_scale = 1.0 / K

            hidden_state = scale_gradient(hidden_state, 0.5)
            
            # Targets
            z, u, pi, mask = target_values[:,k], target_rewards[:,k], target_policies[:,k], masks[:,k]
            
            value_loss = mse(value, z)
            reward_loss = mse(reward, u)
            policy_loss = mse(policy_logits, pi) #tf.linalg.matmul(pi, policy_logits, transpose_a=True, transpose_b=False)
            combined_loss = value_loss + reward_loss + policy_loss

            loss += scale_gradient(combined_loss, gradient_scale)
            
            # Metric logging for tensorboard
            value_loss_metric(value_loss)
            reward_loss_metric(reward_loss)
            policy_loss_metric(policy_loss)
        
        total_loss_metric(loss)
#         total_reward()
        
        # Todo: Eventually we want to use keras layer regularization or AdamW
        for weights in network.get_weights():
            loss += weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in weights]) # sum of l2 norm of weight matrices - also consider tf.norm
    
    # Is there a cleaner way to implement this?
    f_grad = f_tape.gradient(loss, network.f.trainable_variables)
    g_grad = g_tape.gradient(loss, network.g.trainable_variables)
    h_grad = h_tape.gradient(loss, network.h.trainable_variables)
    optimizer.apply_gradients(zip(f_grad, network.f.trainable_variables))
    optimizer.apply_gradients(zip(g_grad, network.g.trainable_variables))
    optimizer.apply_gradients(zip(h_grad, network.h.trainable_variables))


# Use categorical/softmax cross-entropy loss rather than binary/logistic
# Value and reward are non-logits, actions are logits
def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return cce_loss(prediction, target)


def test_network(config: MuZeroConfig, storage: SharedStorage,
                                    replay_buffer: ReplayBuffer):
    while ray.get(replay_buffer.get_buffer_size.remote()) < 1:
        time.sleep(1)
    network = storage.latest_network()    
    
    for i in range(1):
        ray_id = replay_buffer.sample_batch.remote(50, config.td_steps)
        batch = ray.get(ray_id)
        test_step(config, network, batch)
        
def test_step(config:MuZeroConfig, network: Network, batch):
    """
    Not finished
    This should test 
    We should have 2 functions: one to plot a batch trajectory, another to run inference on a brand new/unseen game in realtime
    """
    observations, actions, target_values, target_rewards, target_policies, masks = batch
    loss = 0
    value_losses = []
    reward_losses = []
    policy_losses = []
    total_losses =  []
    total_rewards = []
              
    K = actions.shape[1] # seqlen
         
    env = gym.make(config.gym_env_name)
    o = env.reset()
    render = plt.imshow(env.render(mode='rgb_array'))
    for k in range(K):
        render.set_data(env.render(mode='rgb_array')) # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)
        
        if k==0:
            # Initial step, from the real observation.
            value, reward, policy_logits, hidden_state = network.initial_inference(observations)
            gradient_scale = 1.0
        else:
            # All following steps
            value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, actions[:,k])
            gradient_scale = 1.0 / K
            
        # Select action greedily
        greedy_action = np.argmax(polic_logits)
        
        # Step in the environment
        o, r, done, _ = env.step(a)
            
        hidden_state = scale_gradient(hidden_state, 0.5)

        # Targets
        z, u, pi, mask = target_values[:,k], target_rewards[:,k], target_policies[:,k], masks[:,k]

        value_loss = mse(value, z)
        reward_loss = mse(reward, u)
        policy_loss = mse(policy_logits, pi) 
        combined_loss = value_loss + reward_loss + policy_loss

        loss += scale_gradient(combined_loss, gradient_scale)
        
        value_losses.append(value_loss)
        reward_losses.append(reward_loss)
        policy_losses.append(policy_loss)
        total_losses.append(loss)
        total_rewards.append(reward.numpy())
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    # Metric logging for tensorboard
    value_loss_metric(value_loss)
    reward_loss_metric(reward_loss)
    policy_loss_metric(policy_loss)

    total_loss_metric(loss)
    