import sys
import os
from time import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ml.calculations import action
from ml.networks import PointerNetwork, Critic
from ml.enviroment import ml_model, combined_feature_size
from structures.structures import Solution
import time
import torch

pointer_network, critic, env = None, None, None

def load_model():
    global pointer_network, critic, env

    pointer_network = PointerNetwork(input_dim=combined_feature_size)
    critic = Critic(combined_feature_size)

    pointer_network.load_state_dict(torch.load('./ml/pointer_network.pth'))
    critic.load_state_dict(torch.load('./ml/critic_network.pth'))

    # Sets the networks in evaluation mode
    pointer_network.eval()
    critic.eval()

    env = ml_model()

def ml_solution(solution: Solution):
    '''Function for the machine learning model to iteratively improve the solution''' 
    env.set_solution(solution)

    time_start = time.time()
    while time.time() - time_start < 55:
        current_state_features = env.get_observation()
        visit1, visit2, before_after, _, _, _ = action(pointer_network, current_state_features)
        env.take_action(visit1, visit2, before_after)
    
    return env.return_solution()