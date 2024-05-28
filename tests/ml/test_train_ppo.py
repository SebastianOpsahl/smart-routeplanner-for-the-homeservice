import unittest
from tests.ml.test_train_ppo import compute_gae
import torch
import torch.nn.functional as F
from torch import nn

# Mock Actor and Critic models for testing
class MockActor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MockActor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MockCritic(nn.Module):
    def __init__(self, input_dim):
        super(MockCritic, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Mock data creation for testing the functions
input_dim = 4
output_dim = 3
num_steps = 5

actor = MockActor(input_dim, output_dim)
critic = MockCritic(input_dim)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=0.001)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.001)

# Creating states, actions, rewards, and terminals
states = torch.randn(num_steps, input_dim)
actions = torch.randint(0, output_dim, (num_steps,))
rewards = torch.randn(num_steps)
is_terminals = torch.tensor([0] * (num_steps - 1) + [1], dtype=torch.float32)

# Computing expected values using critic for demonstration
values = critic(states).squeeze().detach()

# Assuming the next value is zero for simplicity
next_value = 0

# Mock RolloutBuffer - Normally, this would store data during environment interaction
class MockRolloutBuffer:
    def __init__(self):
        self.states = [states[i] for i in range(num_steps)]
        self.actions = [actions[i] for i in range(num_steps)]
        self.rewards = [rewards[i] for i in range(num_steps)]
        self.is_terminals = [is_terminals[i] for i in range(num_steps)]
        # Log probabilities are not calculated here but would typically be stored during data collection
        self.logprobs = [torch.tensor(0) for _ in range(num_steps)]  # Placeholder

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.logprobs.clear()

# Create a RolloutBuffer instance
rollout_buffer = MockRolloutBuffer()

class test_train_ppo(unittest.TestCase):
    def test_compute_gae():
        returns, advantages = compute_gae(next_value, rewards, is_terminals, values)
        print("Returns and advantages: ", returns, advantages)