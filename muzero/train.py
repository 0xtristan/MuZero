##################################
####### Part 2: Training #########

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.compat.v1.train import get_global_step
import ray
from tqdm import tqdm
import time
import datetime
import matplotlib.pyplot as plt
from IPython import display
import pdb

from .config import MuZeroConfig
from .storage import SharedStorage, ReplayBuffer
from .models import Network, Network_CNN, Network_FC, scalar_to_support, support_to_scalar
from .selfplay import play_game


#@ray.remote
def train_network(config: MuZeroConfig, storage: SharedStorage,
                                    replay_buffer: ReplayBuffer):
    while ray.get(replay_buffer.get_buffer_size.remote()) < 1:
        time.sleep(1)
    # Network at start of training will have shit weights
    network = Network_FC(config) if config.model_type == "fc" else Network_CNN(config) # Network()
#     network_weights = ray.get(storage.latest_weights.remote())
#     network.set_weights(network_weights)
    
    learning_rate = config.lr_init * config.lr_decay_rate**(0)#get_global_step() / config.lr_decay_steps)
    optimizer = Adam(learning_rate, config.momentum)  
    progbar = tf.keras.utils.Progbar(config.training_steps, verbose=1, stateful_metrics=None, unit_name='step')
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for i in range(config.training_steps): #tqdm(range(config.training_steps), desc='Training iter'):

        if i % config.checkpoint_interval == 0:
            storage.save_weights.remote(i, network.get_weights())
        ray_id = replay_buffer.sample_batch.remote(config.num_unroll_steps, config.td_steps)
        batch = ray.get(ray_id)
        vl, rl, pl, wrl, tl, fg, gg, hg, v_pred, r_pred, p_pred, v_targ, r_targ, p_targ = train_step(optimizer, network, batch, config.weight_decay)
        progbar.update(i, values=[('Value loss', vl),
                                  ('Reward loss', rl),
                                  ('Policy loss', pl),
                                  ('Weight Reg loss', wrl),
                                  ('Total loss', tl),
                                 ])
        if i%20==0:
            _,total_reward = play_game(config, network, render=False)
            
        with train_summary_writer.as_default():
            tf.summary.scalar('1. Losses/Value loss', vl, step=i)
            tf.summary.scalar('1. Losses/Reward loss', rl, step=i)
            tf.summary.scalar('1. Losses/Policy loss', pl, step=i)
            tf.summary.scalar('1. Losses/Weight Reg loss', wrl, step=i)
            tf.summary.scalar('1. Losses/Total loss', tl, step=i)
            tf.summary.histogram("Grads/F first layer gradients", fg[0], step=i, buckets=None)
            tf.summary.histogram("Grads/G first layer gradients", gg[0], step=i, buckets=None)
            tf.summary.histogram("Grads/H first layer gradients", hg[0], step=i, buckets=None)
            tf.summary.histogram("Grads/F final layer gradients", fg[-1], step=i, buckets=None)
            tf.summary.histogram("Grads/G final layer gradients", gg[-1], step=i, buckets=None)
            tf.summary.histogram("Grads/H final layer gradients", hg[-1], step=i, buckets=None)
            tf.summary.scalar('2. Predictions/Value prediction mean', v_pred, step=i)
            tf.summary.scalar('2. Predictions/Reward prediction mean', r_pred, step=i)
            tf.summary.histogram('2. Predictions/Policy prediction dist', p_pred, step=i)
            tf.summary.scalar('3. Targets/Value target mean', v_targ, step=i)
            tf.summary.scalar('3. Targets/Reward target mean', r_targ, step=i)
            tf.summary.histogram('3. Targets/Policy target dist', p_targ, step=i)
            if i%20==0:
                tf.summary.scalar('Reward', total_reward, step=i)

    storage.save_weights.remote(config.training_steps, network.get_weights())


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
    value_loss_metric = tf.keras.metrics.Mean()
    reward_loss_metric = tf.keras.metrics.Mean()
    policy_loss_metric = tf.keras.metrics.Mean()
#     weight_reg_loss_metric = tf.keras.metrics.Mean()
#     total_loss_metric = tf.keras.metrics.Mean()

    value_pred_mean = tf.keras.metrics.Mean()
    reward_pred_mean = tf.keras.metrics.Mean()
    policy_pred_dist = []
    
    value_target_mean = tf.keras.metrics.Mean()
    reward_target_mean = tf.keras.metrics.Mean()
    policy_target_dist = []

    observations, actions, target_values, target_rewards, target_policies, masks = batch
    with tf.GradientTape() as f_tape, tf.GradientTape() as g_tape, tf.GradientTape() as h_tape:
        loss = 0
        K = actions.shape[1] # seqlen
        for k in range(K):
            if k==0:
                # Initial step, from the real observation.
                value, reward, policy_logits, hidden_state = network.initial_inference(observations, convert_to_scalar=False)
                gradient_scale = 1.0
            else:
                # All following steps
                value, reward, policy_logits, hidden_state = network.recurrent_inference(hidden_state, actions[:,k], convert_to_scalar = False)
                gradient_scale = 1.0 / (K-1)

            hidden_state = scale_gradient(hidden_state, 0.5)
            
            # Targets
            z, u, pi, mask = target_values[:,k], target_rewards[:,k], target_policies[:,k], masks[:,k]
            
            value_loss = ce_loss(value, scalar_to_support(z, network.value_support_size), mask)
            reward_loss = ce_loss(reward, scalar_to_support(u, network.value_support_size), mask)
            policy_loss = ce_loss(policy_logits, pi, mask) #tf.linalg.matmul(pi, policy_logits, transpose_a=True, transpose_b=False)
            combined_loss = 1.0*value_loss + 1.0*reward_loss + 1.0*policy_loss
#             pdb.set_trace()

            loss += scale_gradient(combined_loss, gradient_scale)
            
            # Metric logging for tensorboard
            value_loss_metric(value_loss)
            reward_loss_metric(reward_loss)
            policy_loss_metric(policy_loss)
            
            value_pred_mean(support_to_scalar(value, network.value_support_size)*mask)
            reward_pred_mean(support_to_scalar(reward, network.value_support_size)*mask)
            policy_pred_dist.append(tf.nn.softmax(policy_logits)*mask)
            value_target_mean(z*mask)
            reward_target_mean(u*mask)
            policy_target_dist.append(pi*mask)
        
#         total_loss_metric(loss)
#         total_reward()
        
        # Todo: Eventually we want to use keras layer regularization or AdamW
        weight_reg_loss = 0
        for weights in network.get_weights():
            weight_reg_loss += weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in weights]) # sum of l2 norm of weight matrices - also consider tf.norm
#         weight_reg_loss_metric(weight_reg_loss)
        loss += weight_reg_loss
    
    # Is there a cleaner way to implement this?
    f_grad = f_tape.gradient(loss, network.f.trainable_variables)
    g_grad = g_tape.gradient(loss, network.g.trainable_variables)
    h_grad = h_tape.gradient(loss, network.h.trainable_variables)
    optimizer.apply_gradients(zip(f_grad, network.f.trainable_variables))
    optimizer.apply_gradients(zip(g_grad, network.g.trainable_variables))
    optimizer.apply_gradients(zip(h_grad, network.h.trainable_variables))
    # We ought to average or sum these losses - otherwise reward loss is 0 at the end lol
    return value_loss_metric.result(), reward_loss_metric.result(), policy_loss_metric.result(), weight_reg_loss, loss, f_grad, g_grad, h_grad, value_pred_mean.result(), reward_pred_mean.result(), policy_pred_dist, value_target_mean.result(), reward_target_mean.result(), policy_target_dist


# Use categorical/softmax cross-entropy loss rather than binary/logistic
# Value and reward are non-logits, actions are logits
def mse_loss(y_pred, y_true, mask) -> float:
    # MSE in board games, cross entropy between categorical values in Atari. 
    return tf.reduce_mean(tf.math.squared_difference(y_pred, y_true)*mask, axis=[0,1])

def ce_loss(y_pred, y_true, mask) -> float:
    # MSE in board games, cross entropy between categorical values in Atari. 
#     return tf.reduce_mean(-y_true*tf.nn.log_softmax(y_pred, axis=None)*mask, axis=[0,1])
#     pdb.set_trace()
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)*tf.squeeze(mask))
#     return tf.reduce_sum(-y_true*tf.math.log(y_pred)*mask, axis=[0,1])

