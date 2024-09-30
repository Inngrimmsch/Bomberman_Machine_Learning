import os
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.nn.functional import conv2d, max_pool2d, softmax

from skimage.morphology import flood, flood_fill

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.w_s1 = self.model['ws1']
        self.as1 = self.model['as1']
        self.w_s2 = self.model['ws2']
        self.as2 = self.model['as2']
        self.w_so = self.model['wso']
        self.conv_w1 = self.model['cw1']
        self.w_t1 = self.model['wt1']
        self.at1 = self.model['at1']
        self.w_t2 = self.model['wt2']
        self.at2 = self.model['at2']
        self.w_to = self.model['wto']
        self.w_c1 = self.model['wc1']
        self.ac1 = self.model['ac1']
        self.w_c2 = self.model['wc2']
        self.ac2 = self.model['ac2']
        self.w_co = self.model['wco']
        self.w_d1 = self.model['wd1']
        self.ad1 = self.model['ad1']
        self.w_d2 = self.model['wd2']
        self.ad2 = self.model['ad2']
        self.w_d3 = self.model['wd3']
        self.ad3 = self.model['ad3']
        self.w_do = self.model['wdo']
        

    if self.train:
        self.logger.info("Training model.")
        self.epsilon = 0.05
        self.p_dropout_input = 0.0
        self.p_dropout_hidden = 0.05 #0.2
        
    else:
        self.p_dropout_input = 0
        self.p_dropout_hidden = 0
    self.cumulative_reward = 0
    




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    (self.x, self.y) = game_state['self'][3]
    change = np.array([1, 1, 1, 1, 1, 1])
    if game_state['self'][2] == False:
        change[5] = 0


    field = game_state['field']
    for (n,s,b,(x,y)) in game_state['others']:
        field[x][y] = 1 
    for ((x,y), t) in game_state['bombs']:
        field[x][y] = 1
    if field[self.x][self.y-1] != 0:
        change[0] = 0

    if field[self.x][self.y+1] != 0:
        change[2] = 0

    if field[self.x+1][self.y] != 0:
        change[1] = 0

    if field[self.x-1][self.y] != 0:
        change[3] = 0
    #print("change",change)

    # todo Exploration vs exploitation
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        prob = [.2, .2, .2, .2, .1, .1]
        prob = prob*change
        prob = prob/(prob.sum())
        #print(prob)
        out = np.random.choice(ACTIONS, p=prob)
        #print("action", out)

        return out


    self.logger.debug("Querying model for action.")
    self.Q_table = Neural_network_model(self, game_state)
    #print("Q_table", self.Q_table)
    change = torch.tensor(change, dtype=torch.float32)
    prob = self.Q_table
    prob = prob/(max(prob))
    prob = softmax(prob, dim=0)
    prob = prob*change
    #print("softmax1", prob)
    prob = softmax(prob, dim=0)
    prob = prob.detach().cpu().numpy()
    prob = prob.squeeze()
    #print("softmax", prob)
    out = np.random.choice(ACTIONS, p=prob)
    #print("action", out)
    return out                                           


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    (self.x, self.y) = game_state['self'][3]

    #this is to make sure that the faetures are always 7x7 even if the agent is close to the edge
    start_vis_x = self.x-3
    end_vis_x = self.x+4
    start_vis_y = self.y-3
    end_vis_y = self.y+4

    #here we make sure we only copy parts of arrays that exist in explosion map etc (define the borders)
    start_x = max(start_vis_x, 0)
    end_x = min(end_vis_x, game_state['field'].shape[0])  # field and explosion map have the same shape so this works for both
    start_y = max(start_vis_y, 0)
    end_y = min(end_vis_y, game_state['field'].shape[1])
    #here im not sure if my slicing is correct because it does get a little messy
    slice_start_x = max(start_x-start_vis_x, 0)
    slice_end_x = slice_start_x+end_x-start_x
    slice_start_y = max(start_y-start_vis_y, 0) 
    slice_end_y = slice_start_y+end_y-start_y

    expl = game_state['explosion_map']  #this technically doesnt account for the fact that walls block explosions, but as this is just the case for the bombs before they explode i dont think this i gonna be a big problem
    for ((x,y), t) in game_state['bombs']:      
        expl[max(x-(3-t),0):min(x+(3-t), expl.shape[0]), y] += 5    #max and min ensure that the danger from bombs doesnt get calculatred for tiles outside of "explosion map"
        expl[x, max(y-(3-t),0):min(y+(3-t),expl.shape[1])] += 5

    #flood filling the free_tiles around the agent so a non reachable free tile doesnt work for the safety check
    reachable_field = np.zeros((7, 7))
    reachable_field = reachable_field -1 #i did thsi so "out of bounds parts of the map are not seen as free tiles"
    reachable_field[slice_start_x:slice_end_x, slice_start_y:slice_end_y]= game_state['field'][start_x:end_x, start_y:end_y]
    reachable_field = (reachable_field == 0).astype(int)
    reachable_field[3][3] = 1 #this is so the floodfill works when theres a bomb below the agent
    #print("reach", reachable_field)
    flooded_field = flood_fill(reachable_field, (3, 3), 2, connectivity=1)
    #print("flooded field", flooded_field)


    close_danger_map = np.zeros((7, 7))
    close_danger_map[slice_start_x:slice_end_x, slice_start_y:slice_end_y]= expl[start_x:end_x, start_y:end_y].astype(int)
    if close_danger_map[3][3] !=0: #if on danger
        self.danger = 1
        steps_to_safety = np.array([10,10,10,10,10])
        #oben
        above = [(2, 2), (3, 2), (4, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]
        for (x, y) in above:
            if close_danger_map[x][y] == 0 and flooded_field[x][y]== 2:
                distance = x+y
                if steps_to_safety[0] > distance:
                    steps_to_safety[0] = distance
        #rechts
        right =  [(4, 2), (4, 3), (4, 4), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
        for (x, y) in right:
            if close_danger_map[x][y] == 0 and flooded_field[x][y] ==2:
                distance = x+y
                if steps_to_safety[1] > distance:
                    steps_to_safety[1] = distance
        #unten
        below = [(2, 4), (3, 4), (4, 4), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]
        for (x, y) in below:
            if close_danger_map[x][y] == 0 and flooded_field[x][y]==2:
                distance = x+y
                if steps_to_safety[2] > distance:
                    steps_to_safety[2] = distance
        #links
        left = [(2, 2), (2, 3), (2, 4), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
        for (x, y) in left:
            if close_danger_map[x][y] == 0 and flooded_field[x][y]==2:
                distance = x+y
                if steps_to_safety[3] > distance:
                    steps_to_safety[3] = distance
    else: #if in safety
        self.danger = 0
        steps_to_safety = np.zeros(5)
        #oben
        above = [(2, 2), (3, 2), (4, 2), (3, 1)]
        for (x, y) in above:
            if close_danger_map[x][y] != 0 and flooded_field[x][y]==2:
                steps_to_safety[0] += 1
        #rechts
        right = [(4, 2), (4, 3), (4, 4), (5, 3)]
        for (x, y) in right:
            if close_danger_map[x][y] != 0 and flooded_field[x][y]==2:
                steps_to_safety[1] += 1
        #unten
        below = [(2, 4), (3, 4), (4, 4), (3, 5)]
        for (x, y) in below:
            if close_danger_map[x][y] != 0 and flooded_field[x][y]==2:
                steps_to_safety[2] += 1
        #links
        left = [(2, 2), (2, 3), (2, 4),(1, 3)]
        for (x, y) in left:
            if close_danger_map[x][y] != 0 and flooded_field[x][y]==2:
                steps_to_safety[3] += 1



    #bomb_targets_feature
    start_vis_x = self.x-6
    end_vis_x = self.x+7   #plus 5 because when slicing the end is excluded
    start_vis_y = self.y-6
    end_vis_y = self.y+7

    #here we make sure we only copy parts of arrays that exist in explosion map etc (define the borders)
    start_x = max(start_vis_x, 0)
    end_x = min(end_vis_x, game_state['field'].shape[0])  # field and explosion map have the same shape so this works for both
    start_y = max(start_vis_y, 0)
    end_y = min(end_vis_y, game_state['field'].shape[1])
    #here im not sure if my slicing is correct because it does get a little messy
    slice_start_x = max(start_x-start_vis_x, 0)
    slice_end_x = slice_start_x+end_x-start_x
    slice_start_y = max(start_y-start_vis_y, 0) 
    slice_end_y = slice_start_y+end_y-start_y

    field_around_agent = np.zeros((13, 13))
    field_around_agent[slice_start_x:slice_end_x, slice_start_y:slice_end_y]= game_state['field'][start_x:end_x, start_y:end_y]
    for (n,s,b,(x,y)) in game_state['others']:                 
        if x in range (start_x, end_x) and y in range (start_y, end_y):
            field_around_agent[x-start_vis_x, y-start_vis_y] = 1

    bomb_targets = np.zeros((5, 5))
    for x in range(3, 8):
        for y in range(3, 8):
            for xe in range(x-3, x+4):
                if field_around_agent[xe][y] == - 1:
                    break
                elif field_around_agent[xe][y] == 0:
                    bomb_targets[x-3, y-3]  += 1
            for ye in range(y-3, y+4): #doesnt matter that we count xy twice cause there cant be a crate on our position
                    if field_around_agent[x][ye] == - 1:
                        break
                    elif field_around_agent[x][ye] == 0:
                        bomb_targets[x-3, y-3]  += 1
    self.bomb_score = bomb_targets[2, 2]

    #coin_density_feature
    above_left = [(2, 2), (3, 2), (1, 1), (2, 1), (3, 1), (0, 0), (1, 0), (2, 0), (3, 0)]
    al = 0
    above_right = [(3, 2), (4, 2), (3, 1), (4, 1), (5, 1), (3, 0), (4, 0), (5, 0), (6, 0)]
    ar = 0
    right_up =  [(4, 2), (4, 3), (5, 1), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), (6, 3)]
    ru = 0
    right_down =  [(4, 3), (4, 4), (5, 3), (5, 4), (5, 5), (6, 3), (6, 4), (6, 5), (6, 6)]
    rd = 0
    below_left = [(2, 4), (3, 4), (1, 5), (2, 5), (3, 5), (0, 6), (1, 6), (2, 6), (3, 6)]
    bl = 0
    below_right = [(3, 4), (4, 4), (5, 4), (3, 5), (4, 5), (5, 5), (3, 6), (4, 6), (5, 6)]
    br = 0
    left_up = [(3, 4), (4, 4), (3, 5), (4, 5), (5, 5), (3, 6), (4, 6), (5, 6), (6, 6)]
    lu = 0
    left_down = [(2, 3), (2, 4), (1, 3), (1, 4), (1, 5), (0, 3), (0, 4), (0, 5), (0, 6)]
    ld = 0
    for (x,y) in game_state['coins']:
        if (x-start_vis_x, y-start_vis_y) in above_left:
            al += 1
        if (x-start_vis_x, y-start_vis_y) in above_right:
            ar += 1
        if (x-start_vis_x, y-start_vis_y) in right_up:
            ru += 1
        if (x-start_vis_x, y-start_vis_y) in right_down:
            rd += 1
        if (x-start_vis_x, y-start_vis_y) in below_left:
            bl += 1
        if (x-start_vis_x, y-start_vis_y) in below_right:
            br += 1
        if (x-start_vis_x, y-start_vis_y) in left_up:
            lu += 1
        if (x-start_vis_x, y-start_vis_y) in left_down:
            ld += 1
    coin_denstity_feature = np.zeros(4)
    coin_denstity_feature[0] = max(al, ar)
    coin_denstity_feature[1] = max(ru, rd)
    coin_denstity_feature[2] = max(bl, br)
    coin_denstity_feature[3] = max(lu, ld)
    
    self.best_coin_move = np.argmax(coin_denstity_feature)



    steps_to_safety = torch.tensor(steps_to_safety, dtype=torch.float32)
    bomb_targets = torch.tensor(bomb_targets, dtype=torch.float32).unsqueeze(0)
    coin_denstity_feature = torch.tensor(coin_denstity_feature, dtype=torch.float32)
    
    return steps_to_safety, bomb_targets, coin_denstity_feature

def dropout(x, p_drop):
    if not 0 <= p_drop < 1:
        raise ValueError(f"Dropout Probability must be between 0 and 1.") 
    if p_drop == 0.0:
        return x
    
    binomial_mask = torch.rand(x.size()) < p_drop
    x_dropped = x.clone()
    x_dropped[binomial_mask] = 0 
    return x_dropped / (1.0 - p_drop)

def PreLu(x, a):
    return torch.where(x > 0, x, a*x)

def Leaky_ReLu(x):
    return torch.where(x > 0, x, 0.1*x)

def ReLu(x):
    return torch.max(torch.zeros_like(x), x)


def Neural_network_model (self, game_state):   #input states !! 
    #feature extraction

    safety_feat, target_feat, coin_feat = state_to_features(self, game_state)
    safety_feat = dropout(safety_feat, self.p_dropout_input)
    target_feat = dropout(target_feat, self.p_dropout_input)
    coin_feat = dropout(coin_feat, self.p_dropout_input)

    #safety subnet
    s1 = PreLu(safety_feat @ self.w_s1, self.as1)
    dropout_s1 = dropout(s1, self.p_dropout_hidden)
    s2 = PreLu(dropout_s1 @ self.w_s2, self.as2)
    dropout_s2 = dropout(s2, self.p_dropout_hidden)
    so = dropout_s2@self.w_so

    #target subnet
    leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    conv1 = conv2d(target_feat, self.conv_w1)
    conv1 = leaky_relu(conv1)
    drop1 = dropout(conv1, self.p_dropout_hidden)

    x = torch.reshape(drop1, (1, -1))

    t1 = PreLu(x @ self.w_t1, self.at1)
    dropout_t1 = dropout(t1, self.p_dropout_hidden)
    t2 = PreLu(dropout_t1 @ self.w_t2, self.at2)
    dropout_t2 = dropout(t2, self.p_dropout_hidden)
    to = dropout_t2@self.w_to
    to = to.squeeze(0)
    #coin subnet
    c1 = PreLu(coin_feat @ self.w_c1, self.ac1)
    dropout_c1 = dropout(c1, self.p_dropout_hidden)
    c2 = PreLu(dropout_c1 @ self.w_c2, self.ac2)
    dropout_c2 = dropout(c2, self.p_dropout_hidden)
    co = dropout_c2@self.w_co

    #decision net
    #print("so",so)
    #print("to", to)
    #print("co", co)
    decision_input = torch.cat((so, to, co), dim = 0)

    d1 = PreLu(decision_input @ self.w_d1, self.ad1)
    dropout_d1 = dropout(d1, self.p_dropout_hidden)
    d2 = PreLu(dropout_d1 @ self.w_d2, self.ad2)
    dropout_d2 = dropout(d2, self.p_dropout_hidden)
    d3 = PreLu(dropout_d2 @ self.w_d3, self.ad3)
    dropout_d3 = dropout(d3, self.p_dropout_hidden)
    #print("dropout", dropout_d3)
    do = dropout_d3@self.w_do

    return do