from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import os

import events as e
from .callbacks import state_to_features

import torch
import torch.optim as optim

import torchvision.transforms as transforms
from torch.nn.functional import conv2d, max_pool2d

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


# Events
X = "X"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.alpha = 0.2
    self.gamma = 0.9
    # Example: Setup an array that will note transition tuples

    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    if not os.path.isfile("my-saved-model.pt"):     #all the shapes need to be changed!!!
        self.conv_w1 = init_weights((32, 6, 3, 3))
        self.conv_w2 = init_weights((64, 32, 3, 3))
        self.w_h1= init_weights((64, 128))
        self.w_o = init_weights((128, 6))
        self.a = init_alpha(128,)

    self.optimizer = optim.Adam([self.conv_w1, self.conv_w2, self.w_h1, self.w_o, self.a], lr = self.alpha)


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

    #these two also dont really make sense anymore because i dont really search for targets...
    if X_condition(old_game_state, new_game_state):
        events.append(X)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
     
    #update according to temporan difference learning  (loss) (Q(s,a)=Q(s,a)+alpha*(....))  (same thing also in end of round method)
    #neural network training with loss = TD

    Q_approx = self.Q_table
    



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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    #here the Q_update from "game events accured" needs to be pasted with the part with discounted future rewards (self.gamma) removed

    # Store the model
    self.model = {'wc1': self.conv_w1, 'wc2': self.conv_w2, 'w1': self.w_h1, 'wo': self.w_o, 'a': self.a}
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
        e.CRATE_DESTROYED: 0.25*(3+np.exp(-step*0.2)),
        e.COIN_COLLECTED: 0.5*(1+np.exp(-step*0.2)),
        e.KILLED_OPPONENT: 5,
        e.GOT_KILLED: -5,
        e.KILLED_SELF: -10,
        e.INVALID_ACTION: -3,
        #e.MOVE_TO_TARGET: 0.2,
        #e.MOVE_AWAY_FROM_TARGET: -0.2,
        e.SURVIDED_ROUND: 5,
        #e.blocked
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum





def X_condition(old_game_state: dict, new_game_state: dict) -> bool:
    if ...:
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