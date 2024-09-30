from collections import namedtuple, deque

import numpy as np
import pickle
import random
from typing import List

import events as e
from .callbacks import state_to_features, ACTIONS, rotate_right, reflect_horizontal_four,\
                            reflect_vertical_four

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'state_key', 'tracer'))

#### python main.py play --no-gui --agents rasmus04 rule_based_agent rule_based_agent rule_based_agent --train 1

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 50  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

##

MOVED_AWAY_FROM_DANGER = 'MOVED_AWAY_FROM_DANGER'
MOVED_TO_COINS = 'MOVED_TO_COINS'

def q_update(self, old_game_state):

    action_to_index = {action: i for i, action in enumerate(ACTIONS)}

    ## initialising update parameters
    count = 0
    reward_sum = 0
    temporal_difference = 6

    if old_game_state['step'] < 6:
        temporal_difference = old_game_state['step']

    update_action = action_to_index[self.transitions[old_game_state['step']-temporal_difference].action]
    last_action = action_to_index[self.transitions[old_game_state['step']-1].action]

    update_tracer = self.transitions[old_game_state['step']-temporal_difference].tracer
    last_tracer = self.transitions[old_game_state['step']-1].tracer

    ## calculating the update
    old_reward_slice = [self.transitions[i].reward for i in \
                        range(old_game_state['step']-temporal_difference,old_game_state['step']-2)]
    
    for old_reward in old_reward_slice:
        reward_sum = reward_sum + old_reward * np.power(self.gamma,count)
        count +=1

    update_table = self.q_table[self.transitions[old_game_state['step']-temporal_difference].state_key]

    ## Correcting for possible state match in different orientation
    if update_tracer < 4:
        update_action = (update_action + update_tracer)%4
    elif update_tracer > 3:
        if (update_tracer-4) % 2 == 0:
            if update_action == 1 or update_action == 3:
                update_action = (update_action + 2)%4
            update_action = (update_action + update_tracer-4)%4
        elif (update_tracer-4) % 2 == 1:
            if update_action == 0 or update_action == 2:
                update_action = (update_action + 2)%4
            update_action = (update_action + update_tracer-4)%4

    if last_tracer < 4:
        last_action = (last_action + last_tracer)%4
    elif last_tracer > 3:
        if (last_tracer-4) % 2 == 0:
            if last_action == 1 or last_action == 3:
                last_action = (last_action + 2)%4
            last_action = (last_action + last_tracer-4)%4
        elif (last_tracer-4) % 2 == 1:
            if last_action == 0 or last_action == 2:
                last_action = (last_action + 2)%4
            last_action = (last_action + last_tracer-4)%4

    reward_sum = reward_sum + self.q_table[self.transitions[old_game_state['step']-1].state_key]\
        [last_action] * np.power(self.gamma,count)
    
    updated_table = update_table
    updated_table[update_action] =\
        update_table[update_action] + self.alpha * (reward_sum - update_table[update_action])

    self.q_table[self.transitions[old_game_state['step']-temporal_difference].state_key]\
        = updated_table
    return 

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.gamma = 0.99

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):

    action_to_index = {action: i for i, action in enumerate(ACTIONS)}

    last_action = action_to_index[self_action]

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, reward_from_events(old_game_state, self, events),self.current_location,self.tracer))

    dir1, dir2, tile, center = self.transitions[old_game_state['step']-1].state

    if last_action < 4 and dir2[last_action] == 1:
        events.append(MOVED_AWAY_FROM_DANGER)

    if last_action < 4 and dir1[last_action] >= 1:
        events.append(MOVED_TO_COINS)

    if random.random() > self.use_state:
        q_update(self,old_game_state)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]): 

    self.game_number +=1
    
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, reward_from_events(last_game_state, self, events),self.current_location,self.tracer))

    action_to_index = {action: i for i, action in enumerate(ACTIONS)}

    last_action = action_to_index[last_action]

    dir1, dir2, tile, center = self.transitions[last_game_state['step']-1].state

    if last_action < 4 and dir2[last_action] == 1:
        events.append(MOVED_AWAY_FROM_DANGER)

    if last_action < 4 and dir1[last_action] >= 1:
        events.append(MOVED_TO_COINS)
    
    q_update(self,last_game_state)

    if self.game_number % 10000 == 0 and self.game_number > 1:
        print('After ' + str(self.game_number) + ' games, ' +\
              str(len(self.q_table)) + ' states have been explored')
    
    # Store the model
    self.model = {'q_table': self.q_table, 'random_prob': self.random_prob, 'alpha': self.alpha, 'game_number': self.game_number}
    with open("modelno0.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(old_game_state, self, events: List[str]) -> int:

    game_rewards = {
        e.CRATE_DESTROYED: 0,
        e.COIN_COLLECTED: 6,
        e.KILLED_OPPONENT: 0,
        e.GOT_KILLED: -10,
        e.COIN_FOUND: 0,
        e.KILLED_SELF: -10,
        e.MOVED_LEFT: 2,
        e.MOVED_RIGHT: 2,
        e.MOVED_UP: 2,
        e.MOVED_DOWN: 2,
        e.WAITED: -4,
        e.INVALID_ACTION: -5,
        e.SURVIVED_ROUND: 0,
        e.BOMB_DROPPED: -8,
        e.MOVED_AWAY_FROM_DANGER: 2,
        e.MOVED_TO_COINS: 2,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum