import os
import pickle
import random

import numpy as np
from scipy.special import softmax

ACTIONS = ['RIGHT', 'UP', 'LEFT', 'DOWN', 'WAIT', 'BOMB']

def setup(self):
    self.current_location = ''
    self.random_prob = 0
    self.alpha = 0.1
    self.use_state = 0.4
    self.last_action = 5
    if os.path.isfile("modelno0.pt"):
        with open("modelno0.pt", "rb") as file:
            self.model = pickle.load(file)
        self.game_number = self.model['game_number']
        self.q_table = self.model['q_table']
    else:
        self.game_number = 0
        self.q_table = {}
    self.random_prob = self.random_prob * np.exp(-self.game_number/300000)
    self.alpha = self.alpha * np.exp(-self.game_number/1000000)
        
def act(self, game_state: dict) -> str:
    action_to_index = {action: i for i, action in enumerate(ACTIONS)}
    self.logger.debug("Querying model for action.")
    dir1, dir2, tile, center = state_to_features(game_state)
    ## Make the model 'commit' to search for coins in a certain direction
    if self.last_action < 4:
        if self.last_action < 2:
            dir1[self.last_action + 2] = 0
        else:
            dir1[self.last_action - 2] = 0
    #####################################################################
    exists, location, tracer = q_rotational_symmetry(self, dir1, dir2, tile, center)
    self.current_location = location
    if not exists: ## FOR OUTSIDE TRAINING ADD SIMILARITY FACTOR TO NOT MAKE RANDOM CHOICE HERE
        initial_guess = np.array([0,0,0,0,0,0])
        dir1_array = np.array(dir1)
        dir2_array = np.array(dir2)
        for i in range(4):
            initial_guess[i] = dir1_array[i] + 2*dir2_array[i]
        if np.abs(tile[3]) != 0:
            initial_guess[0] = -10
        if np.abs(tile[1]) != 0:
            initial_guess[1] = -10
        if np.abs(tile[7]) != 0:
            initial_guess[2] = -10
        if np.abs(tile[5]) != 0:
            initial_guess[3] = -10
        self.q_table[location] = initial_guess

    if self.train and random.random() < self.random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        self.last_action = action_to_index[choice]
        return choice
    
    q_values = list(self.q_table[location])
    movement_q_values = q_values[0:4]
    actual_q_values = movement_q_values

    ## print('TRACER: ' + str(tracer))
    ## print('Q_VALUES_BEFORE: ' + str(q_values))

    ## Correcting for possible state match in different orientation
    if tracer < 4:
        actual_q_values = rotate_right(movement_q_values, -tracer)
    elif tracer > 3:
        if tracer-4 % 2 == 0:
            actual_q_values = rotate_right(reflect_horizontal_four(movement_q_values),-(tracer-4))
        elif tracer-4 % 2 == 1:
            actual_q_values = rotate_right(reflect_vertical_four(movement_q_values),-(tracer-4))
    else:
        actual_q_values = movement_q_values
    q_values = list([actual_q_values[0],actual_q_values[1],actual_q_values[2],actual_q_values[3],q_values[4],q_values[5]])

    ## safe_state_name = ''
    ## for i in range(4):
    ##    safe_state_name = safe_state_name + str(int(dir1[i]))
    ## for i in range(4):
    ##    safe_state_name = safe_state_name + str(int(dir2[i]))
    ## for i in range(8):
    ##    safe_state_name = safe_state_name + str(int(tile[i]))
    ## safe_state_name = safe_state_name + str(int(center))

    choice = np.random.choice(ACTIONS, p=softmax(q_values))
    ## print('CHOICE: ' + choice)
    ## print('Q_VALUES: ' + str(q_values))
    ## print('STATE_KEY: ' + self.current_location)
    ## print('STATE: ' + safe_state_name)
    ## actions = ['RIGHT', 'UP', 'LEFT', 'DOWN', 'WAIT', 'BOMB']
    self.last_action = action_to_index[choice]
    return choice

def bound_check(game_state: dict) -> np.array:
    ## make sure we do not try to observe things outside of the map
    shape_vector = [4,4,4,4]
    if game_state['self'][3][0] < 4:
        shape_vector[2] = game_state['self'][3][0]
    if game_state['self'][3][0] +4 > 16:
        shape_vector[0] = 17 - game_state['self'][3][0]
    if game_state['self'][3][1] < 4:
        shape_vector[1] = game_state['self'][3][1]
    if game_state['self'][3][1] +4 > 16:
        shape_vector[3] = 17 - game_state['self'][3][1]
    return shape_vector

def extraction(game_state: dict, shape_vector) -> np.array:
    ## space crates stuff
    space_crates_feature = np.zeros((3,3))
    ## danger stuff
    bombs_feature = np.zeros((3,3))
    explosion_feature = np.zeros((3,3))
    ## coins stuff
    coins_feature = np.zeros((3,3))
    ## players stuff
    player_feature = np.zeros((3,3))
    ## general information
    x = game_state['self'][3][0]
    y = game_state['self'][3][1]
    player_scan = np.zeros((17,17))
    coins_scan = np.zeros((17,17))
    bombs_scan = np.zeros((17,17))
    ## extracting space_crates and players first
    space_crates_feature = np.transpose(game_state['field'][x-1:x+2,y-1:y+2]) ## 3x3 map
    for item in game_state['others']:
        player_scan[item[3][0],item[3][1]] = 1
    player_feature = np.transpose(player_scan[x-1:x+2,y-1:y+2]) ## 3x3 map
    ## extracting the rest
    explosion_scan = np.transpose(game_state['explosion_map']\
        [x-shape_vector[2]:x+shape_vector[0]+1,y-shape_vector[1]:y+shape_vector[3]+1]) ## 9x9 map
    for item in game_state['coins']:
        coins_scan[item[0],item[1]] = 1 ## 17x17 map
    for item in game_state['bombs']:
        bombs_scan[item[0][0],item[0][1]] = 5-item[1] ## 17x17 map
    ## converting the rest into features
    for i, j in np.ndindex(coins_feature.shape):
        coins_feature[j,i] = np.sum(coins_scan[i:i+shape_vector[0]+shape_vector[2]-1,\
                                               j:j+shape_vector[1]+shape_vector[3]-1]) ## 3x3 map
    for i, j in np.ndindex(bombs_feature.shape):
        bombs_feature[j,i] = np.max(np.concatenate((bombs_scan\
            [i:i+shape_vector[0]+shape_vector[2]-1,j+shape_vector[1]-1],\
            bombs_scan[i+shape_vector[2]-1,j:j+shape_vector[1]+shape_vector[3]-1]))) ## 3x3 map
    for i, j in np.ndindex(explosion_feature.shape):
        explosion_feature[i,j] = np.max(np.concatenate((explosion_scan\
            [i:i+shape_vector[0]+shape_vector[2]-1,j+shape_vector[1]-1],\
            explosion_scan[i+shape_vector[2]-1,j:j+shape_vector[1]+shape_vector[3]-1]))) ## 3x3 map
    ## initialise state description variables
    possible_movements = list(space_crates_feature)
    coin_density = [0,0,0,0]
    highest_coin_density = [0,0,0,0]
    danger = [0,0,0,0]
    lowest_danger = [0,0,0,0]
    ## calculate state description variables based on the features of the respective game state
    # calculate directional coin density
    coin_density[0] = np.sum(coins_feature[2,0:3])
    coin_density[1] = np.sum(coins_feature[0:3,0])
    coin_density[2] = np.sum(coins_feature[0,0:3])
    coin_density[3] = np.sum(coins_feature[0:3,2])
    for i in range(4):
        if coin_density[i] == np.max(coin_density):
            highest_coin_density[i] = 1 ## 1x4 vector
    # calculate directional danger estimate
    danger[0] = np.sum(bombs_feature[2,0:3]) + np.sum(explosion_feature[2,0:3])
    danger[1] = np.sum(bombs_feature[0:3,0]) + np.sum(explosion_feature[0:3,0])
    danger[2] = np.sum(bombs_feature[0,0:3]) + np.sum(explosion_feature[0,0:3])
    danger[3] = np.sum(bombs_feature[0:3,2]) + np.sum(explosion_feature[0:3,2])
    for i in range(4):
        if danger[i] == np.min(danger):
            lowest_danger[i] = 1 ## 1x4 vector
    # for possible moves consider as well explosions, about to explode bombs and players
    possible_movements = possible_movements - player_feature
    for i,j in np.ndindex(possible_movements.shape):
        if possible_movements[i,j] == 0:
            if explosion_feature[i,j] > 0 or bombs_feature[i,j] == 5:
                possible_movements[i,j] = -1 ## 3x3 map
    ## create 1-dim output which will serve for rotational symmetry checks and state naming
    surroundings = flatten_list(possible_movements)
    ## swap center of fov to the back of the array to make rotational symmetry check easier
    temp = surroundings[4]
    for i in range(4):
        surroundings[4+i] = surroundings[5+i] ## 1x8 vector
    surroundings[8] = temp ## 1x1 vector
    return highest_coin_density, lowest_danger, surroundings[0:8], surroundings[8]

def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    ## checking for close boundaries of the board
    shaping = bound_check(game_state)

    ## returning all features
    return extraction(game_state,shaping)

def rotate_right(arr, steps):
    return arr[-steps:] + arr[:-steps]

def reflect_horizontal(tiles): ## reflects along y-axis
    return [tiles[5], tiles[6], tiles[7], tiles[3], tiles[4], tiles[0], tiles[1], tiles[2]]

def reflect_vertical(tiles): ## reflects along x-axis
    return [tiles[2], tiles[1], tiles[0], tiles[4], tiles[3], tiles[7], tiles[6], tiles[5]]

def reflect_horizontal_four(input):
    return list([input[0],input[3],input[2],input[1]])

def reflect_vertical_four(input):
    return list([input[2],input[1],input[0],input[3]])

def get_rotations_and_reflections(dir1, dir2, tiles, center):
    variations = []
    # Generate rotational variants (0°, 90°, 180°, 270°)
    for i in range(4):
        rotated_dir1 = rotate_right(dir1, i)
        rotated_dir2 = rotate_right(dir2, i)
        rotated_tiles = rotate_right(tiles, i * 2)  # rotate by 2 positions for each 90° rotation
        variations.append(list((rotated_dir1, rotated_dir2, rotated_tiles)))

    # Generate reflected variants for each rotation
    for i in range(4):
        rotated_dir1, rotated_dir2, rotated_tiles = variations[i]

        # Reflect horizontally
        variations.append(list((reflect_horizontal_four(rotated_dir1), reflect_horizontal_four(rotated_dir2), reflect_horizontal(rotated_tiles))))

        # Reflect vertically
        variations.append(list((reflect_vertical_four(rotated_dir1), reflect_vertical_four(rotated_dir2), reflect_vertical(rotated_tiles))))

    for i in range(len(variations)):
        variations[i].append(center)
        variations[i].append(i)
        
    return variations

def q_rotational_symmetry(self,input0,input1,input2,input3):

    # Get all possible transformations of the current state
    possible_states = get_rotations_and_reflections(input0, input1, input2, input3)
    keysList = list(self.q_table.keys())

    safe_state_name = ''
    for i in range(4):
        safe_state_name = safe_state_name + str(int(input0[i]))
    for i in range(4):
        safe_state_name = safe_state_name + str(int(input1[i]))
    for i in range(8):
        safe_state_name = safe_state_name + str(int(input2[i]))
    safe_state_name = safe_state_name + str(int(input3))

    # Check if any of the transformed states match an already saved state
    for state in possible_states:
        state_name = ''
        for i in range(4):
            state_name = state_name + str(int(state[0][i]))
        for i in range(4):
            state_name = state_name + str(int(state[1][i]))
        for i in range(8):
            state_name = state_name + str(int(state[2][i]))
        state_name = state_name + str(int(state[3]))
        for keyname in keysList:
            if keyname == state_name:
                return True, state_name, state[4]
    return False, safe_state_name, 0

def flatten_list(xss):
    return [x for xs in xss for x in xs]