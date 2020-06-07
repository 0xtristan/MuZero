from typing import List
import gym
import cv2
import tensorflow as tf
import numpy as np
import pdb
from scipy.signal import lfilter

from .mcts_numpy import Node, ActionHistory
from .config import MuZeroConfig

ENVS = {
    'breakout': 'Breakout-v0',
    'pong': 'Pong-v0',
    'cartpole': 'CartPole-v0',
}

class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self, config: MuZeroConfig):
        self.gym_env = gym.make(config.gym_env_name)
        self.prepro = config.gym_env_name=='Breakout-v0'
        initial_state = self.prepro(self.gym_env.reset()) if self.prepro else self.gym_env.reset()
        self.obs_history = [initial_state]
        self.done = False
        
    def step(self, action: int):
        obs, reward, self.done, info = self.gym_env.step(action)
        if self.prepro:
            obs = self.prepro(obs)
        self.obs_history.append(obs)
        return float(reward)
    
    def legal_actions(self):
        """Env specific rules for legality of moves
        TODO: if at wall don't allow movement into the wall"""
        return [a for a in range(self.gym_env.action_space.n)]

    def prepro(self, obs, size=(80,80)):
        """Crop, resize, B&W"""
        p_obs = obs[25:195,:,0] / 255 # crop and normalise to [0,1]
        return cv2.resize(p_obs, size, interpolation=cv2.INTER_NEAREST) # resize

    def get_obs(self, start:int, end:int=None):
        return self.obs_history[max(start,0):end]
    
    
class Game(object):
    """A single episode of interaction with the environment. (One trajectory)"""
    def __init__(self, config: MuZeroConfig):
        self.env = Environment(config)
        self.history = [] # actual actions a
        self.rewards = [] # observed rewards u
        self.child_visits = [] # search tree action distributions pi
        self.root_values = [] # values ν
        self.action_space_size = config.action_space_size
        self.gamma = config.discount
        self.config = config

    def prepare_to_save(self):
        # Clean up for saving, no need to save the env object - sims can be expensive.
        self.env.gym_env.close()
        self.env.gym_env = None
        # And I'm now about to do ground truth values, and try overtrain them
        self.ground_truth_values = discount_cumsum(self.rewards, self.gamma)
        # note this is a little weird in 'stay alive' style games, as the value decreases as the game goes on
        # in goal completion style (-1 until 0 when goal completed, value increases as you get closer to achieving the goal).

    def terminal(self) -> bool:
        if self.env.done:
            self.env.gym_env.close()
        return self.env.done
    
    def legal_actions(self) -> List[int]:
        return self.env.legal_actions()
    
    def apply(self, action: int):
        reward = self.env.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        return reward
        
    def store_search_statistics(self, root: Node):
        """Stores the MCTS search value of node and search policy (visits ratio of children)"""
        sum_visits = np.sum([child.visit_count for child in root.children.values()])
        # search policy π = [0.1,0.5,0.4] probability over children
        self.child_visits.append(
            np.array([
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in range(self.action_space_size)
            ], dtype='float32')
        )
        # search value ν
        self.root_values.append(root.value())
    
    def make_image(self, t: int, feat_history_len:int = 32) -> tf.Tensor:
        """Observation at chosen position w/ history"""
        # Game specific feature planes
        # For Atari we have the 32 most recent RGB frames at resolution 96x96
        # Instead I use 80x80x1 B&W
        if t==-1: t=len(self.env.obs_history)-1
        frames = self.env.get_obs(t-feat_history_len+1, t+2) # We want 32 frames up to and including t
        # Cast to tensor and add dummy batch dim
        # Todo: figure out how to stack RGB images - i.e. colour & time dimensions
        frame_tensor = tf.convert_to_tensor(np.stack(frames,axis=-1))
        # Pad out sequences with insufficient history
        # I believe there is a bug here because the get_batch() gets a -1  pad value
        padding_size = feat_history_len-frame_tensor.shape[-1]
        # Todo: should this be [0, padding_size]?? Too tired to test
        if len(frame_tensor.shape)==2:
#             padded_frames = tf.pad(frame_tensor, paddings=[[0, 0], [0, padding_size]], constant_values=0)
            # Fuck it let's just forget the history - this is a shitty hack
            padded_frames = tf.convert_to_tensor(self.env.obs_history[-1], dtype=tf.float32) # (1,4)
        else:
            padded_frames = tf.pad(frame_tensor, paddings=[[0, 0], [0, 0], [0, padding_size]], constant_values=0)
#         padded_frames = tf.expand_dims(padded_frames, 0) # dummy batch dim
        return padded_frames
    
    # This uses TD estimate for v
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    # Todo: We can swap this for more PPO style version later
    def compute_target_value(self, current_index:int, td:int):

        bootstrap_index = current_index + td 
        # If our TD lookahead is still before the end of the game, the update with that 
        # future game state value estimate ν_{t+N}
        if bootstrap_index < len(self.root_values):
            # γ^N*ν_{t+N}
            value =  self.root_values[bootstrap_index] * self.gamma**td
        else:
            value = 0

        # Rest of the TD value estimate from observed rewards: u_{t+1} + ... + γ^{N-1} u_{t+n}
        for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
            value += reward * self.gamma**i  
        return value
    
    def make_target(self, t: int, K: int, td: int):
        """
        (value,reward,policy) target for each unroll step t to t+K
        This is taken from actuals results of the game (to be stored in replay buffer)
        Uses TD learning to calculate value target via n step bootstrap, see above
        """
        # Returns target tuple (value, reward, policy) i.e. (z,u,pi)
        # K=5
        # s0 (z0,u0,pi0,a0) - s1 (z1,u1,pi1,a1) - s2 (z2,u2,pi2,a2)
        # returns len K+1 sequences
        target_values, target_rewards, target_policies = [], [], []
        # We use a boolean masking vector to indicate the end of a sequence (à la NLP)
        mask, policy_mask = [], []
        # K + 1 iterations
        for current_index in range(t, t + K + 1):
            ## Value Target z_{t+K} ##

            value = self.compute_target_value(current_index, td)
                
            ## Reward u_{t+K} and Action π_{t+K} Targets ##
            
            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > t and current_index <= len(self.rewards):
                # self.rewards[i] is the reward received after the ith state
                # self.rewards[i-1] is therefore the most recently received reward
                last_reward = self.rewards[current_index - 1]
            else:
                # For i=t and i=t+K+1
                # Be careful because we have to set r=0 for first target (not first gamestep) due to initial inference
                last_reward = 0

            if current_index < len(self.root_values):
                target_values.append(value)
                target_rewards.append(last_reward)
                target_policies.append(self.child_visits[current_index])
                policy_mask.append(1)
            else:
                # States past the end are treated as absorbing states
                target_values.append(0)
                target_rewards.append(last_reward) # 0
                # Uniform policy: 1/len(actions) for all actions
                # Todo: is this a valid thing to do? @Sholto review
                # This is probability targets not logits
                # uniform_policy = [1/self.config.action_space_size
                #                   for _ in range(self.config.action_space_size)]
                dummy_policy = np.array([1/self.config.action_space_size for _ in range(self.config.action_space_size)], dtype='float32')
                # dummy_policy = np.array([-1 for _ in range(self.config.action_space_size)],dtype='float32')
                target_policies.append(dummy_policy)
                policy_mask.append(0)

            if current_index <= len(self.root_values):
                mask.append(1)
            else:
                mask.append(0)

            # v = [v0,v1,v2,0,0]
            # r = [0,r0,r1,r2,0]
            # p = [p0,p1,p2,_,_]
            # m = [1,1,1,1,0]
            # pm = [1,1,1,0,0]
            # a = [_,a0,a1,a3,_] - first index is initial_inference and thus no action
                
        return target_values, target_rewards, target_policies, mask, policy_mask
    
    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]