import sys
import os
from time import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from ml.networks import PointerNetwork, Critic
from ml.calculations import action
from ml.enviroment import combined_feature_size
from ml.enviroment import ml_model
from structures.structures import Solution
import time
#from fastapi.applications import LifespanEvents

# Globals for the loaded models and enviroment
pointer_network, critic, env = None, None, None

#@app.lifecycle.on(LifespanEvents.STARTUP)
def load_model():
    global pointer_network, critic, env

    pointer_network = PointerNetwork(input_dim=combined_feature_size)
    critic = Critic(combined_feature_size)

    pointer_network.load_state_dict(torch.load('./ml_model/pointer_network.pth'))
    critic.load_state_dict(torch.load('./ml_model/critic_network.pth'))

    # Sets the networks in evaluation mode
    pointer_network.eval()
    critic.eval()

    env = ml_model()

def predict(solution: Solution): 
    env.set_solution(solution)

    time_start = time.time()
    while time.time() - time_start < 55:
        current_state_features = env.get_observation()
        visit1, visit2, before_after, _, _, _ = action(pointer_network, current_state_features)
        env.take_action(visit1, visit2, before_after)
    
    return env.return_solution()