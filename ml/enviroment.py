import gym
from gym import spaces
from genetic.genetic import fitness, feasible
from copy import deepcopy
import numpy as np
import torch

visit_feature_size = 4
global_feature_size = 4
combined_feature_size = visit_feature_size + global_feature_size

class ml_model(gym.Env):
    def __init__(self):
        super(ml_model, self).__init__()
        '''Initializes the enviroment with steps per episode'''
        self.current_step = 0
        self.max_steps = 50

    def set_solution(self, solution):
        '''sets the solution as an enviroment. Creates the action and observation space'''
        self.solution = solution

        # mapping from values to representation in neural network
        self.visit_id_to_index, self.index_to_visit_id = self.initialize_mappings()
        self.num_visits = len(self.visit_id_to_index)

        self.action_space = spaces.MultiDiscrete([self.num_visits, self.num_visits])
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_visits * visit_feature_size + global_feature_size,), dtype=np.float32)
        
        # returns the initial observation for the new solution
        return self.get_observation()

    def step(self, visit1_idx, visit2_idx, before_after):
        '''A step consists of putting visit1 either before or after visit2'''
        self.current_step += 1

        old_score = fitness(self.solution.routes, self.solution.matrix)

        visit1_id = self.index_to_visit_id[visit1_idx.item()]
        visit2_id = self.index_to_visit_id[visit2_idx.item()]

        solution_copy = deepcopy(self.solution)
        solution_copy.move_visit(visit1_id, visit2_id, before_after)

        is_feasible = feasible(solution_copy.routes, solution_copy.shift_time, solution_copy.break_time, solution_copy.matrix)
        new_score = fitness(solution_copy.routes, solution_copy.matrix)     

        reward = self.reward(new_score, old_score, is_feasible)
        if is_feasible:
            self.solution.move_visit(visit1_id, visit2_id, before_after)
        
        done = self.current_step >= self.max_steps

        return self.get_observation(), reward, done
    
    def take_action(self, visit1_idx, visit2_idx, before_after):
        '''Function to take the action based on the trained model, this will be used in production'''
        visit1_id = self.index_to_visit_id[visit1_idx.item()]
        visit2_id = self.index_to_visit_id[visit2_idx.item()]

        solution_copy = deepcopy(self.solution)
        solution_copy.move_visit(visit1_id, visit2_id, before_after)

        is_feasible = feasible(solution_copy.routes, solution_copy.shift_time, solution_copy.break_time, solution_copy.matrix)
        if is_feasible:
            new_score = fitness(solution_copy.routes, solution_copy.matrix)
            old_score = fitness(self.solution.routes, self.solution.matrix)
            if new_score < old_score:
                self.solution.move_visit(visit1_id, visit2_id, before_after)

    def return_solution(self):
        return self.solution

    def reset(self):
        '''Resets state for new observations from the start'''
        self.current_step = 0
        self.set_solution(self.solution)
        return self.get_observation()

    def get_observation(self):
        '''Get's an observation of the enviroment to use for the neural networks'''
        num_visits = sum(len(route.visits) for route in self.solution.routes)
        feature_size = 8

        observations_np = np.zeros((num_visits, feature_size), dtype=np.float32)
        
        idx = 0
        for route in self.solution.routes:
            for visit in route.visits:
                observations_np[idx, 0] = visit.start_time
                observations_np[idx, 1] = visit.end_time
                observations_np[idx, 2] = visit.task_time
                observations_np[idx, 3] = visit.double_staffed
                observations_np[idx, 4] = self.solution.shift_time[0]
                observations_np[idx, 5] = self.solution.shift_time[1]
                observations_np[idx, 6] = self.solution.break_time[0]
                observations_np[idx, 7] = self.solution.break_time[1]
                idx += 1

        # adds batch dimension
        final_tensor = torch.from_numpy(observations_np).unsqueeze(0)

        return final_tensor

    def initialize_mappings(self):
        '''Helper function to map neural network representations of visits to real world representation'''
        visit_id_to_index = {}
        index_to_visit_id = {}
        index = 0
        for route in self.solution.routes:
            for visit in route.visits:
                visit_id_to_index[visit.visit_id] = index
                index_to_visit_id[index] = visit.visit_id
                index += 1
        return visit_id_to_index, index_to_visit_id

    def reward(self, new_score: np.float32, old_score: np.float32, isFeasible: bool):
        '''Gives reward based on performance. None for unfeasible solutins as this should be 
        discouraged at all costs. But slightly bad steps shouldn't be punished to hard.'''
        if not isFeasible:
            return np.float32(0)

        estimated_max_score = 40
        diff = new_score - old_score
        normalized_diff = diff / estimated_max_score

        if diff > 0:
            reward = np.float32(0.6 + 0.5 * min(normalized_diff, 1))
        elif diff < 0:
            reward = np.float32(0.1 * (1 + normalized_diff))
        else:
            # no change in performance give slight rewarded
            reward = np.float32(0.1)
        
        return np.float32(reward)

class RolloutBuffer:
    '''Buffer to store observation'''
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        '''Clears buffer of observations'''
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def add(self, visit1, visit2, before_after, logprob_1, logprob_2, log_prob_ba, state, reward, done):
        '''Adds the observations to the observation buffer'''
        self.states.append(torch.squeeze(state))
        # use of long due to actions being int placement of the visits
        self.actions.append(torch.tensor([visit1.item(), visit2.item(), before_after.item()], dtype=torch.long))
        self.logprobs.append(torch.stack([logprob_1, logprob_2, log_prob_ba.flatten()]))
        self.rewards.append(torch.tensor(reward))
        self.is_terminals.append(torch.tensor(1.0 if done else 0.0, dtype=torch.float32))

    def get_full_batch(self):
        '''Returns observations from the buffer'''
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.logprobs),
            torch.stack(self.rewards),
            torch.stack(self.is_terminals)
        )