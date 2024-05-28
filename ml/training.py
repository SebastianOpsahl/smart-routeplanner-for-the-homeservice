import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ml.enviroment import ml_model, RolloutBuffer, combined_feature_size
from ml.calculations import retrieve_solutions, action, update_policy
from ml.networks import PointerNetwork, Critic
import torch

# retrieve_solution was a function to retrieve test data used for the project
solutions = retrieve_solutions()
env = ml_model()

# sets the sizes of the network based on the fixed values (sizes of features per vists)
# since both networks are adaptable to the variable nature of how many visits
pointer_network = PointerNetwork(input_dim=combined_feature_size)
critic = Critic(combined_feature_size)

epochs = 10
rollout_buffer = RolloutBuffer()

# choses the Adam optimizer and sends with the parameters from the networks
optimizer_pointer_network = torch.optim.Adam(pointer_network.parameters(), lr=0.001)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.001)

# treshhold before policy update
threshold = 20
steps = 0

for solution in solutions:
    env.set_solution(solution)
    for episode in range(450):
        obs = env.reset()
        state = torch.FloatTensor(obs)
        done = False
        while not done:
            # gets observation of the current state
            current_state_features = env.get_observation()

            # returns action to take the step in the enviroment
            action1, action2, before_after, log_prob1, log_prob2, log_prob_ba = action(pointer_network, current_state_features)
            # the step function in your environment needs to accept before_after
            next_state, reward, done = env.step(action1, action2, before_after)

            # stores data in buffer
            # store new action details in the buffer including the before_after boolean and its log probability
            rollout_buffer.add(action1, action2, before_after, log_prob1, log_prob2, log_prob_ba, state, reward, done)

            # updates state
            state = torch.FloatTensor(next_state) 

            steps += 1
            # checks if we have reached the update threshold
            if steps >= threshold:
                # performs PPO update using the accumulated data
                update_policy(pointer_network, critic, optimizer_pointer_network, optimizer_critic, rollout_buffer, epochs)
                steps = 0

torch.save(pointer_network.state_dict(), './pointer_network-2.pth')
torch.save(critic.state_dict(), './critic_network-2.pth')

print("Finished", flush=True)
