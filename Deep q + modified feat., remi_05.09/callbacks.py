import os
import pickle
import random

import numpy as np

import torch
import torch.optim as optim

import torchvision.transforms as transforms
from torch.nn.functional import conv2d, max_pool2d, softmax


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

        
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Training model.")

        self.dropout_input = 0.05
        self.dropout_hidden = 0.05
        self.epsilon = 0.1
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        
        self.dropout_input = 0
        self.dropout_hidden = 0

    self.conv_w1 = self.model['wc1']
    self.conv_w2 = self.model['wc2']
    self.w_h1 = self.model['w1']
    self.wo = self.model['wo']
    self.a = self.model['a']




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    features = state_to_features(self, game_state)
    self.Q_table = model (features, self.conv_w1, self.conv_w2, self.w_h1, self.w_o, self.a, self.p_dropout_input, self.p_dropout_hidden)
    Prob = softmax(self.Q_table)
    return np.random.choice(ACTIONS, p=Prob)                                                  


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
    
    self.x = game_state['self'][3][0]
    self.y = game_state['self'][3][1]

    #more or less "one"-hot encoding (more values than one and zero in the danger map)

    #"danger map" 7x7 

    expl = game_state['explosion_map']  #this technically doesnt account for the fact that walls block explosions, but as this is just the case for the bombs before they explode i dont think this i gonna be a big problem
    for ((x,y), t) in game_state['bombs']:      
        expl[np.max(x-(3-t),0):np.min(x+(3-t), expl.shape[0]), y] += 1    #np.max and min ensure that the danger from bombs doesnt get calculatred for tiles outside of "explosion map"
        expl[x, np.max(y-(3-t),0):np.min(y+(3-t),expl.shape[1])] += 1

    close_danger_map = expl[self.x-3:self.x+4, self.y-3:self.y+4]   #exception am rand (?)


    #only parts of the field to make the feature space managable and only focused on important parts

    #im not sure if changing the original state of "field" into one-hot encoding is necessary or helpful (more dimensions but maybe clearer for the agent)
    field_around_agent = game_state['field'][self.x-3:self.x+4, self.y-3:self.y+4]

    walls = (field_around_agent == -1).astype(int)

    free_tile = (field_around_agent == 0).astype(int)

    crate = (field_around_agent == 1).astype(int)


    coins_around_agent = np.zeros((7, 7))
    for (x,y) in game_state['coins']: 
        if x in range (self.x-3, self.x+4) and y in range (self.y-3, self.y+4):
            coins_around_agent[x-self.x+3, y-self.y+3] = 1  #+3 to make the indices non negative 


    close_others = np.zeros((7, 7))
    for (n,s,b,(x,y)) in game_state['others']:
        if x in range (self.x-3, self.x+4) and y in range (self.y-3, self.y+4):
            close_others[x-self.x+3, y-self.y+3] = 1


    features = np.stack((walls, free_tile, crate, coins_around_agent, close_others, close_danger_map), axis=2)  #7x7x6 feature 

    return features

def dropout(x, p_drop):
    if not 0 <= p_drop < 1:
        raise ValueError(f"Dropout Probability must be between 0 and 1.") 
    if p_drop == 0.0:
        return x
    
    binomial_mask = torch.rand(x.size()) < p_drop
    x[binomial_mask] = 0 
    return x / (1.0 - p_drop)

def PreLu(x, a):
    return torch.where(x > 0, x, a*x)

def ReLu(x):
    return torch.max(torch.zeros_like(x), x)


def model (self, features, conv_w1, conv_w2, w_h1, w_o,):
    features = dropout(features, self.p_dropout_input)

    conv1 = ReLu(conv2d(features, conv_w1))
    pool1 = max_pool2d(conv1, (2, 2))
    drop1 = dropout(pool1, self.p_dropout_hidden)

    conv2 = ReLu(conv2d(drop1, conv_w2))
    pool2 = max_pool2d(conv2, (2, 2))
    drop2 = dropout(pool2, self.p_dropout_hidden)

    x = torch.reshape(drop2, (batch_size, 64))  #change batch size =(experiences)
    
    h1 = PreLu(x @ w_h1, self.a)
    h1 = dropout(h1, self.p_dropout_hidden)

    out = h1 @ w_o
    return out
