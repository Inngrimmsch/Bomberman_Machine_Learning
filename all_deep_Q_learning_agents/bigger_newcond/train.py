from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import random
import os

import events as e
from .callbacks import state_to_features, Neural_network_model, dropout, PreLu, ReLu, ACTIONS

import torch
import torch.optim as optim

import torchvision.transforms as transforms
from torch.nn.functional import conv2d, max_pool2d

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))


# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...



# Events
MOVED_TO_COIN_DENSITY = "MOVED_TO_COIN_DENSITY"
TAKING_A_CORNER = "TAKING_A_CORNER"
REPEATING_ONE_MOVE = "REPEATING_ONE_MOVE"
BACK_AND_FORTH = "BACK_AND_FORTH"
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
    self.learnprob = 0.1 #theroy said about one step per game so i think this should be okay (0-2 steps in early (strongly randomized games) and maximum about 20)

    self.alpha = 0.3
    self.gamma = 1

    self.wc = 0
    self.ccc = 0 #coin collected counter
    self.iac = 0 #invalid action counter
    self.bfc = 0 #back and forth counter
    self.rmc = 0 #repeating move counter
    self.tcc = 0 #taking corner counter
    # Example: Setup an array that will note transition tuples
    action_to_index = {action: i for i, action in enumerate(ACTIONS)}
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.last_action = None
    if not os.path.isfile("my-saved-model.pt"):     
        self.conv_w1 = init_weights((64, 3, 3, 3))
        self.conv_w2 = init_weights((128, 64, 3, 3))
        self.w_h1= init_weights((3200, 1000))
        self.w_h2 = init_weights((1000, 300))
        self.w_o = init_weights((300, 6))
        self.a1 = init_alpha((1000,))
        self.a2 = init_alpha((300,))



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
    if moved_to_coin_density(self, old_game_state, self_action):
        events.append(MOVED_TO_COIN_DENSITY)
    if "COIN_COLLECTED" in events:
        self.ccc += 1
    if "WAITED" in events:
        self.wc += 1
    if "INVALID_ACTION" in events:
        self.iac += 1
    else:
        self.iac = 0
    if taking_a_corner_condition(self, self_action):
        events.append(TAKING_A_CORNER)
    if back_and_forth_condition(self, self_action):
        events.append(BACK_AND_FORTH)

    if SAFETY_condition(self, old_game_state):
        events.append(SAFETY)
    else:
        events.append(DANGER)
    
    if Moved_to_danger_condition(self, old_game_state, new_game_state):
        events.append(MOVED_INTO_DANGER)
    
    if Moved_to_safety_condition(self, old_game_state, new_game_state):
        events.append(MOVED_INTO_SAFETY)

    if repeating_one_move_condition(self, self_action):
        events.append(REPEATING_ONE_MOVE)

    self.last_action = self_action

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(self, old_game_state), self_action, reward_from_events(old_game_state, self, events)))

    true_reward = reward_from_events(old_game_state, self, events)
    self.cumulative_reward = self.cumulative_reward + true_reward


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
    self.transitions.append(Transition(state_to_features(self, last_game_state), last_action, reward_from_events(last_game_state, self, events)))

    action_to_index = {action: i for i, action in enumerate(ACTIONS)}

    torch.autograd.set_detect_anomaly(True)
    self.optimizer = optim.Adam([self.conv_w1, self.conv_w2, self.w_h1, self.w_h2, self.w_o, self.a1, self.a2], lr = self.alpha)
    
    #6-step td
    for i in range(last_game_state['step']-7):
        if random.random() < self.learnprob:
            action_index = action_to_index[self.transitions[i].action]
            current_Qs = Neural_network_model(self, self.transitions[i].state)
            current_Q = current_Qs[action_index]
            reward_1 = self.transitions[i].reward
            reward_2 = self.transitions[i+1].reward
            reward_3 = self.transitions[i+2].reward
            reward_4 = self.transitions[i+3].reward
            reward_5 = self.transitions[i+4].reward
            reward_6 = self.transitions[i+5].reward
            future_Q = torch.max(Neural_network_model(self, self.transitions[i+6].state))

            squared_loss_function = (reward_1+self.gamma*reward_2+self.gamma**2*reward_3+self.gamma**3*reward_4+self.gamma**4*reward_5+self.gamma**5*reward_6+self.gamma**6*future_Q-current_Q)**2
            self.optimizer.zero_grad()
            squared_loss_function.backward()
            self.optimizer.step()

    #5-step td
    if random.random() < self.learnprob:
        sixth_last_step = last_game_state['step']-6
        action_index = action_to_index[self.transitions[sixth_last_step].action]
        current_Qs = Neural_network_model(self, self.transitions[sixth_last_step].state)
        current_Q = current_Qs[action_index]
        reward_1 = self.transitions[sixth_last_step].reward
        reward_2 = self.transitions[sixth_last_step+1].reward
        reward_3 = self.transitions[sixth_last_step+2].reward
        reward_4 = self.transitions[sixth_last_step+3].reward
        reward_5 = self.transitions[sixth_last_step+4].reward

        future_Q = torch.max(Neural_network_model(self, self.transitions[sixth_last_step+5].state))

        squared_loss_function = (reward_1+self.gamma*reward_2+self.gamma**2*reward_3+self.gamma**3*reward_4+self.gamma**4*reward_5+self.gamma**5*future_Q-current_Q)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()
    
    #4-step td
    if random.random() < self.learnprob:
        fifth_last_step = last_game_state['step']-5
        action_index = action_to_index[self.transitions[fifth_last_step].action]
        current_Qs = Neural_network_model(self, self.transitions[fifth_last_step].state)
        current_Q = current_Qs[action_index]
        reward_1 = self.transitions[fifth_last_step].reward
        reward_2 = self.transitions[fifth_last_step+1].reward
        reward_3 = self.transitions[fifth_last_step+2].reward
        reward_4 = self.transitions[fifth_last_step+3].reward
        future_Q = torch.max(Neural_network_model(self, self.transitions[fifth_last_step+4].state))

        squared_loss_function = (reward_1+self.gamma*reward_2+self.gamma**2*reward_3+self.gamma**3*reward_4+self.gamma**4*future_Q-current_Q)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()

    #3-step td
    if random.random() < self.learnprob:
        fourth_last_step = last_game_state['step']-4
        action_index = action_to_index[self.transitions[fourth_last_step].action]
        current_Qs = Neural_network_model(self, self.transitions[fourth_last_step].state)
        current_Q = current_Qs[action_index]
        reward_1 = self.transitions[fourth_last_step].reward
        reward_2 = self.transitions[fourth_last_step+1].reward
        reward_3 = self.transitions[fourth_last_step+2].reward
        future_Q = torch.max(Neural_network_model(self, self.transitions[fourth_last_step+3].state))

        squared_loss_function = (reward_1+self.gamma*reward_2+self.gamma**2*reward_3+self.gamma**3*future_Q-current_Q)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()

    #2-step td
    if random.random() < self.learnprob:
        third_last_step =last_game_state['step']-3
        action_index = action_to_index[self.transitions[third_last_step].action]
        current_Qs = Neural_network_model(self, self.transitions[third_last_step].state)
        current_Q = current_Qs[action_index]
        reward_1 = self.transitions[third_last_step].reward
        reward_2 = self.transitions[third_last_step+1].reward
        future_Q = torch.max(Neural_network_model(self, self.transitions[third_last_step+2].state))

        squared_loss_function = (reward_1+self.gamma*reward_2+self.gamma**2*future_Q-current_Q)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()

    #1-step td  
    if random.random() < self.learnprob:
        second_last_step = last_game_state['step']-2
        action_index = action_to_index[self.transitions[second_last_step].action]
        current_Qs = Neural_network_model(self, self.transitions[second_last_step].state)
        current_Q = current_Qs[action_index]
        reward_1 = self.transitions[second_last_step].reward
        future_Q = torch.max(Neural_network_model(self, self.transitions[second_last_step+1].state))

        squared_loss_function = (reward_1+self.gamma*future_Q-current_Q)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()

    #last step
    last_step = last_game_state['step']-1 #again zero indexing
    if random.random() < self.learnprob: 
        action_index = action_to_index[self.transitions[last_step].action]
        current_Qs = Neural_network_model(self, self.transitions[last_step].state)
        current_Q = current_Qs[action_index]
        reward_1 = self.transitions[last_step].reward

        squared_loss_function = (reward_1-current_Q)**2
        self.optimizer.zero_grad()
        squared_loss_function.backward()
        self.optimizer.step()

    #txt
    true_reward = reward_from_events(last_game_state, self, events)
    self.cumulative_reward = self.cumulative_reward + true_reward
    print("cumulative reward = ", self.cumulative_reward)
    print("survived until:", last_step)
    with open('training.txt', 'a') as file:
        file.write(f'{self.cumulative_reward}, {last_step}\n')
    self.cumulative_reward = 0
    self.ccc = 0 #coin collected counter
    self.iac = 0 #invalid action counter
    self.bfc = 0 #back and forth counter
    self.rmc = 0 #repeating move counter
    self.tcc = 0 #taking corner counter
    self.wc = 0
    #if self.alpha >0.001:
     #   self.alpha = self.alpha*0.999999
    #if self.epsilon > 0.1:
     #   self.epsilon = self.epsilon*0.99999
    print("epsilon", self.epsilon)

    # Store the model
    self.model = {'wc1': self.conv_w1, 'wc2': self.conv_w2, 'w1': self.w_h1, 'w2': self.w_h2, 'wo': self.w_o, 'a1': self.a1, 'a2': self.a2}
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(old_game_state, self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    step = old_game_state["step"]

    if self.rmc < 6:
        self.rmp = 0
    else:
        self.rmp = self.rmc

    game_rewards = {
        e.SURVIVED_ROUND: 300,
        e.COIN_COLLECTED: 150*self.ccc,
        e.MOVED_TO_COIN_DENSITY: 100,
        e.TAKING_A_CORNER: 50*self.tcc,
        e.KILLED_OPPONENT: 5,#change this when training in classic scenario
        e.CRATE_DESTROYED: 3, #change this when training in classic scenario
        e.COIN_FOUND: 1, #change this when training in classic scenario
        e.MOVED_INTO_SAFETY: 10,
        e.SAFETY: 5,
        e.MOVED_LEFT: 5,
        e.MOVED_RIGHT: 5,
        e.MOVED_UP: 5,
        e.MOVED_DOWN: 5,
        e.DANGER: -5,
        e.MOVED_INTO_DANGER: -10,
        e.INVALID_ACTION: -50*self.iac, #theoretically not crazy bad but i want to get trid of it 
        e.WAITED: -50*self.wc, #obviously waiting isnt inherently bad but i had the problem that agents just didnt move at all
        e.GOT_KILLED: -100,
        e.REPEATING_ONE_MOVE: -50 * self.rmp, #repeating move punishment
        e.BACK_AND_FORTH: -80 * self.bfc,
        e.KILLED_SELF: -1000,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

#in motion cond / stopped cond

def moved_to_coin_density(self, last_game_state, self_action):
    if self_action == ('WAIT' or 'BOMB'):
        return False
    cdu, cdd, cdr, cdl = 0, 0, 0, 0
    (self.x, self.y) = last_game_state['self'][3]
    for (x,y) in last_game_state['coins']: 
        x_rel = x-self.x
        y_rel = y-self.y
        if x_rel in range(-1, 2):
            if y_rel in range(0, 5):
                cdu += 1
            if y_rel in range(-4, 1):
                cdd += 1
        if y_rel in range(-1, 2):
            if x_rel in range(-4, 1):
                cdl += 1
            if x_rel in range(0, 5):
                cdr += 1
    highest_desity = max(cdu, cdd, cdl, cdr)
    if self_action == 'UP' and highest_desity == cdu:
        return True
    if self_action == 'DOWN' and highest_desity == cdd:
        return True
    if self_action == 'LEFT' and highest_desity == cdl:
        return True
    if self_action == 'RIGHT' and highest_desity ==cdr:
        return True

def taking_a_corner_condition(self, self_action):
    if self.last_action == None:
        return False
    elif (self.last_action == 'UP' or self.last_action == 'DOWN') and (self_action == 'RIGHT' or self_action == 'LEFT'):
        self.tcc += 1
        return True
    elif (self.last_action == 'RIGHT' or self.last_action == 'LEFT') and (self_action == 'UP' or self_action == 'DOWN'):
        self.tcc+= 1
        return True


def back_and_forth_condition(self, self_action):
    if self.last_action == None:
        return False
    elif self.last_action == 'UP' and self_action == 'DOWN':
        self.bfc += 1
        return True
    elif self.last_action == 'DOWN' and self_action == 'UP':
        self.bfc += 1
        return True
    elif self.last_action == 'LEFT' and self_action == 'RIGHT':
        self.bfc += 1
        return True
    elif self.last_action == 'RIGHT' and self_action == 'LEFT':
        self.bfc += 1
        return True
    else:
        self.bfc = 0

    
def repeating_one_move_condition(self, self_action):
    if self.last_action == self_action:
        self.rmc += 1
    else:
        self.rmc = 0


def SAFETY_condition(self, old_game_state: dict) -> bool:
    features = state_to_features(self, old_game_state)
    if features[0, 2, 4, 4] == 0:
        return True
    else:
        return False
    
def Moved_to_safety_condition(self, old_game_state: dict, new_game_state: dict) -> bool:
    oldfeat = state_to_features(self, old_game_state)
    newfeat = state_to_features(self, new_game_state)
    if newfeat[0, 2, 4, 4] == 0 and oldfeat[0, 2, 4, 4] != 0:
        return True
    else:
        return False
    
def Moved_to_danger_condition(self, old_game_state: dict, new_game_state: dict) -> bool:
    oldfeat = state_to_features(self, old_game_state)
    newfeat = state_to_features(self, new_game_state)
    if newfeat[0, 2, 4, 4] != 0 and oldfeat[0, 2, 4, 4] == 0:
        return True
    else:
        return False


def init_weights(shape):
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size=shape) * std
    w.requires_grad = True
    return w

def init_alpha(shape):
    a = torch.full(size=shape, fill_value=0.25)  #made this 0.25 because i had the problem of dying neurons
    a = a +0.25
    a.requires_grad = True
    return a
