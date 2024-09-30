from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import os

import events as e
from .callbacks import state_to_features, Neural_network_model, dropout, PreLu, ReLu, ACTIONS

import torch
import torch.optim as optim

import torchvision.transforms as transforms
from torch.nn.functional import conv2d, max_pool2d

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...



# Events
SAFETY = "SAFETY"
DANGER = "DANGER"
MOVED_INTO_SAFETY = "MOVED_INTO_SAFETY"
MOVED_INTO_DANGER = "MOVED_INTO_DANGER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.nofeat = 0

    self.alpha = 0.1
    self.gamma = 0.9
    # Example: Setup an array that will note transition tuples
    action_to_index = {action: i for i, action in enumerate(ACTIONS)}
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    if not os.path.isfile("my-saved-model.pt"):     
        self.conv_w1 = init_weights((64, 6, 3, 3))
        self.conv_w2 = init_weights((128, 64, 3, 3))
        self.w_h1= init_weights((3200, 1000))
        self.w_h2 = init_weights((1000, 300))
        self.w_o = init_weights((300, 6))



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards

    if SAFETY_condition(self, old_game_state):
        events.append(SAFETY)
    else:
        events.append(DANGER)
    
    if Moved_to_danger_condition(self, old_game_state, new_game_state):
        events.append(MOVED_INTO_DANGER)
    
    if Moved_to_safety_condition(self, old_game_state, new_game_state):
        events.append(MOVED_INTO_SAFETY)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(old_game_state, self, events)))

    action_to_index = {action: i for i, action in enumerate(ACTIONS)}

    torch.autograd.set_detect_anomaly(True)
    self.optimizer = optim.Adam([self.conv_w1, self.conv_w2, self.w_h1, self.w_h2, self.w_o], lr = self.alpha)
    
    step = old_game_state['step']-1 #zero indexing
    transition = self.transitions[step]
    action_index = action_to_index[transition.action]
    features = transition.state
    next_features = transition.next_state      #find out why this is none sometimes
    true_reward = transition.reward
    self.cumulative_reward = self.cumulative_reward + true_reward

    if next_features is not  None:
        Q_values = Neural_network_model(self, features)
        Q_value = Q_values[action_index]
        next_Q_values = Neural_network_model(self, next_features)
        highest_next_Q = torch.max(next_Q_values)

        squared_loss_function = (true_reward+self.gamma*highest_next_Q-Q_value)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()
    else:
        self.nofeat += 1

    #maybe save the average reward to visualize training progress 





def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]): 
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(last_game_state, self, events)))

    action_to_index = {action: i for i, action in enumerate(ACTIONS)}

    torch.autograd.set_detect_anomaly(True)
    self.optimizer = optim.Adam([self.conv_w1, self.conv_w2, self.w_h1, self.w_h2, self.w_o], lr = self.alpha)
    
    
    last_step = last_game_state['step']-1 #again zero indexing
    action_index = action_to_index[self.transitions[last_step].action]
    features = self.transitions[last_step].state
    true_reward = self.transitions[last_step].reward
    self.cumulative_reward = self.cumulative_reward + true_reward

    Q_values = Neural_network_model(self, features)
    Q_value = Q_values[action_index]
    squared_loss_function = (true_reward-Q_value)**2
    self.optimizer.zero_grad()
    squared_loss_function.backward()
    self.optimizer.step()

    #here the Q_update from "game events accured" needs to be pasted with the part with discounted future rewards (self.gamma) removed

    print("cumulative reward = ", self.cumulative_reward)
    self.cumulative_reward = 0
    print("#features none", self.nofeat)
    self.nofeat = 0
    print("survived until:", last_step)

    with open('rewards.txt', 'a') as file:
        file.write(f'{self.cumulative_reward}, {last_step}\n')
    # Store the model
    self.model = {'wc1': self.conv_w1, 'wc2': self.conv_w2, 'w1': self.w_h1, 'w2': self.w_h2, 'wo': self.w_o}
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(old_game_state, self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    step = old_game_state["step"]

    game_rewards = {
        e.CRATE_DESTROYED: 3,#0.25*(3+np.exp(-step*0.2)),
        e.COIN_COLLECTED: 15,#0.5*(1+np.exp(-step*0.2)),
        e.KILLED_OPPONENT: 20,
        e.GOT_KILLED: -20,
        e.COIN_FOUND: 5,#0.25*(1+np.exp(-step*0.2)),
        e.KILLED_SELF: -30,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.WAITED: 0.5,
        e.INVALID_ACTION: -3,
        e.DANGER: -2,
        e.SAFETY: 2,
        e.MOVED_INTO_SAFETY: 5,
        e.MOVED_INTO_DANGER: -5,
        e.SURVIVED_ROUND: 30,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



def SAFETY_condition(self, old_game_state: dict) -> bool:
    features = state_to_features(self, old_game_state)
    if features[0, 5, 4, 4] == 0:
        return True
    else:
        return False
    
def Moved_to_safety_condition(self, old_game_state: dict, new_game_state: dict) -> bool:
    oldfeat = state_to_features(self, old_game_state)
    newfeat = state_to_features(self, new_game_state)
    if newfeat[0, 5, 4, 4] == 0 and oldfeat[0, 5, 4, 4] != 0:
        return True
    else:
        return False
    
def Moved_to_danger_condition(self, old_game_state: dict, new_game_state: dict) -> bool:
    oldfeat = state_to_features(self, old_game_state)
    newfeat = state_to_features(self, new_game_state)
    if newfeat[0, 5, 4, 4] != 0 and oldfeat[0, 5, 4, 4] == 0:
        return True
    else:
        return False


def init_weights(shape):
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size=shape) * std
    w.requires_grad = True
    return w

def init_alpha(shape):
    a = torch.zeros(size=shape)
    a.requires_grad = True
    return a
