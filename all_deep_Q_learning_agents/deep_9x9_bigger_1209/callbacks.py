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

    if os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
        self.conv_w1 = self.model['wc1']
        self.conv_w2 = self.model['wc2']
        self.w_h1 = self.model['w1']
        self.w_h2 = self.model['w2']
        self.w_o = self.model['wo']

    if self.train:
        self.logger.info("Training model.")

        self.p_dropout_input = 0.0
        self.p_dropout_hidden = 0.0 #0.2
        self.epsilon = 0.0  #0.1
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
    # todo Exploration vs exploitation
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    features = state_to_features(self, game_state)
    self.Q_table = Neural_network_model(self, features)
    Prob = softmax(self.Q_table, dim=0)
    Prob = Prob.detach().cpu().numpy()
    Prob = Prob.squeeze()
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
    
    (self.x, self.y) = game_state['self'][3]

    #this is to make sure that the faetures are always 7x7 even if the agent is close to the edge
    start_vis_x = self.x-4
    end_vis_x = self.x+5   #plus 4 because when slicing the end is excluded
    start_vis_y = self.y-4
    end_vis_y = self.y+5

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

    fov = np.zeros((9, 9))
    #more or less "one"-hot encoding (more values than one and zero in the danger map)

    #"danger map" 9x9 

    expl = game_state['explosion_map']  #this technically doesnt account for the fact that walls block explosions, but as this is just the case for the bombs before they explode i dont think this i gonna be a big problem
    for ((x,y), t) in game_state['bombs']:      
        expl[max(x-(3-t),0):min(x+(3-t), expl.shape[0]), y] +=1    #max and min ensure that the danger from bombs doesnt get calculatred for tiles outside of "explosion map"
        expl[x, max(y-(3-t),0):min(y+(3-t),expl.shape[1])] += 1

    close_danger_map = fov.copy()


    close_danger_map[slice_start_x:slice_end_x, slice_start_y:slice_end_y]= expl[start_x:end_x, start_y:end_y].astype(int)


    #only parts of the field to make the feature space managable and only focused on important parts

    #im not sure if changing the original state of "field" into one-hot encoding is necessary or helpful (more dimensions but maybe clearer for the agent)
    field_around_agent = fov.copy()
    field_around_agent = field_around_agent -1 #this makes it so everything out of buonds will become part of the walls feature
    field_around_agent[slice_start_x:slice_end_x, slice_start_y:slice_end_y]= game_state['field'][start_x:end_x, start_y:end_y]
    
    walls = (field_around_agent == -1).astype(int)
    #maybe add crates to walls to make it a "cant walk there" feature

    free_tile = (field_around_agent == 0).astype(int)

    crate = (field_around_agent == 1).astype(int)

    #no need for the slicing from above because theres never going to be a bomb or enemy out of bound

    coins_around_agent = fov.copy()
    for (x,y) in game_state['coins']: 
        if x in range (start_x, end_x) and y in range (start_y, end_y):  
            coins_around_agent[x-start_vis_x, y-start_vis_y] = 1 

    
    close_others = fov.copy()
    for (n,s,b,(x,y)) in game_state['others']:                 
        if x in range (start_x, end_x) and y in range (start_y, end_y):
            close_others[x-start_vis_x, y-start_vis_y] = 1

    #print("next feat")
    #print("walls", walls)
    #print("free_tiles", free_tile)
    #print("create", crate)
    #print("coins", coins_around_agent)
    #print("close others", close_others)
    #print("close danger", close_danger_map)
    features = np.stack((walls, free_tile, crate, coins_around_agent, close_others, close_danger_map), axis=2)  #7x7x6 feature but 1x6x7x7 needed
    #print("features np.stack", features)
    features = torch.tensor(features, dtype=torch.float32)
    #print("features torch.tensor", features)
    features = features.permute(2, 0, 1).unsqueeze(0)
    #print("feature permute", features)

    return features  

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

def ReLu(x):
    return torch.max(torch.zeros_like(x), x)


def Neural_network_model (self, features):   #input features, not states!!       #added .clone() everywhere before to make sure theres no in-place operation here
    features = dropout(features, self.p_dropout_input).squeeze(0)
    features = features.squeeze(1)
    conv1 = ReLu(conv2d(features, self.conv_w1))
    drop1 = dropout(conv1, self.p_dropout_hidden)

    conv2 = ReLu(conv2d(drop1, self.conv_w2))
    pool2 = max_pool2d(conv2, (2, 2))
    drop2 = dropout(pool2, self.p_dropout_hidden)       #look at this again

    x = torch.reshape(drop2, (1, -1))
    
    h1 = ReLu(x @ self.w_h1)
    dropout_h1 = dropout(h1, self.p_dropout_hidden)

    h2 = ReLu(dropout_h1 @ self.w_h2)
    dropout_h2 = dropout(h2, self.p_dropout_hidden)

    out = torch.matmul(dropout_h2, self.w_o).squeeze(0)
    #print("Features shape:", features.shape),print("Conv1 shape:", conv1.shape),print("Drop1 shape:", drop1.shape),print("Conv2 shape:", conv2.shape),print("Pool2 shape:", pool2.shape),print("Drop2 shape:", drop2.shape),print("Flattened shape:", x.shape),print("h1 shape:", h1.shape),print("Output shape:", out.shape)
    return out
