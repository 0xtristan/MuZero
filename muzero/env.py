from typing import List
import gym
import cv2
import tensorflow as tf
import numpy as np

from .mcts import Node, ActionHistory

ENVS = {
    'breakout': 'Breakout-v0',
    'pong': 'Pong-v0',
    'cartpole': 'CartPole-v0',
}

class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self):
        self.env = gym.make(ENVS['breakout'])
        self.obs_history = [self.prepro(self.env.reset())]
        self.done = False
        
    def step(self, action: int):
        obs, reward, self.done, info = self.env.step(action)
        self.obs_history.append(self.prepro(obs))
        return float(reward)
    
    def terminal(self):
        return self.done
    
    # @tf.function
    def legal_actions(self):
        """Env specific rules for legality of moves
        TODO: if at wall don't allow movement into the wall"""
        return [a for a in range(self.env.action_space.n)]

    def prepro(self, obs, size=(80,80)):
        """Crop, resize, B&W"""
        p_obs = obs[25:195,:,0] / 255 # crop and normalise to [0,1]
        return cv2.resize(p_obs, size, interpolation=cv2.INTER_NEAREST) # resize

    # def prepro(self, obs, size=(80,80)):
    #     return obs
    def get_obs(self, start:int, end:int=None):
        return self.obs_history[max(start,0):end]
    
    
class Game(object):
    """A single episode of interaction with the environment. (One trajectory)"""
    def __init__(self, action_space_size: int, discount: float):
        self.env = Environment()
        self.history = [] # actual actions a
        self.rewards = [] # observed rewards u
        self.child_visits = [] # search tree action distributions pi
        self.root_values = [] # values ν
        self.action_space_size = action_space_size
        self.gamma = discount
    
    def terminal(self) -> bool:
        return self.env.terminal()
    
    def legal_actions(self) -> List[int]:
        return self.env.legal_actions()
    
    def apply(self, action: int):
        reward = self.env.step(action)
        self.rewards.append(reward)
        self.history.append(action)
        
    def store_search_statistics(self, root: Node):
        """Stores the MCTS search value of node and search policy (visits ratio of children)"""
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (index for index in range(self.action_space_size))
        # search policy π = [0.1,0.5,0.4] probability over children
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        # search value ν
        self.root_values.append(root.value())
    
    def make_image(self, t: int, feat_history_len:int = 32) -> tf.Tensor:
        """Observation at chosen position w/ history"""
        # Game specific feature planes
        # For Atari we have the 32 most recent RGB frames at resolution 96x96
        # Instead I use 80x80x1 B&W
        frames = self.env.get_obs(t-feat_history_len+1, t+2) # We want 32 frames up to and including t
        # Cast to tensor and add dummy batch dim
        # Todo: figure out how to stack RGB images - i.e. colour & time dimensions
        frame_tensor = tf.convert_to_tensor(np.stack(frames,axis=-1))
        # If we're missing a channel dim add one
        # if len(frame_tensor.shape)==3:
        #     frame_tensor = frame_tensor.expand_dims(-1) # this is wrong
        # Pad out sequences with insufficient history
        padding_size = feat_history_len-frame_tensor.shape[-1]
        padded_frames = tf.pad(frame_tensor, paddings=[[0, 0], [0, 0], [padding_size, 0]], constant_values=0)
        padded_frames = tf.expand_dims(padded_frames, 0) # dummy batch dim for 4D
        return padded_frames
    
    def make_target(self, t: int, K: int, td: int):
        """
        (value,reward,policy) target for each unroll step t to t+K
        This is taken from actuals results of the game (to be stored in replay buffer)
        Uses TD learning to calculate value target via n step bootstrap, see above
        """
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        # Returns target tuple (value, reward, policy) i.e. (z,u,pi)
        # K=5
        targets = []
        for current_index in range(t, t + K + 1):
            ## Value Target z_{t+K} ##
            
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
                
            ## Reward u_{t+K} and Action π_{t+K} Targets ##
            
            # For simplicity the network always predicts the most recently received
            # reward, even for the initial representation network where we already
            # know this reward.
            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0
            
            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end are treated as absorbing states
                targets.append((0, last_reward, []))
        return targets 
    
    # @tf.function
    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)